import 'dotenv/config';
import express from 'express';
import { Telegraf, Markup } from 'telegraf';
import { OpenAI, toFile } from 'openai';
import { load } from 'cheerio';
import { Pool } from 'pg';

// ===== ENV =====
const TG_TOKEN = process.env.TELEGRAM_BOT_TOKEN;
const OPENAI_KEY = process.env.OPENAI_API_KEY;
const LLM_MODEL = process.env.OPENAI_MODEL || 'gpt-4o-mini';
const ASR_MODEL = process.env.OPENAI_TRANSCRIBE || 'whisper-1';
const PUBLIC_URL = process.env.PUBLIC_URL;
const WEBHOOK_SECRET_PATH = (process.env.WEBHOOK_SECRET_PATH || '/bot').startsWith('/')
  ? (process.env.WEBHOOK_SECRET_PATH || '/bot')
  : `/${process.env.WEBHOOK_SECRET_PATH}`;
const PORT = Number(process.env.PORT || 8080);
const DATABASE_URL = process.env.DATABASE_URL || null;

if (!TG_TOKEN) throw new Error('TELEGRAM_BOT_TOKEN is required');
if (!OPENAI_KEY) throw new Error('OPENAI_API_KEY is required');
if (!PUBLIC_URL) throw new Error('PUBLIC_URL is required for webhooks');

const openai = new OpenAI({ apiKey: OPENAI_KEY });
const bot = new Telegraf(TG_TOKEN);

const app = express();
app.use(express.json());

// ===== простая «оперативная» память в RAM =====
const MAX_TURNS = 8;
const sessions = new Map(); // ключ: `${chat_id}:${user_id}` -> [{role,content}]
const chatLog = new Map();  // ключ: chat_id -> [{role,content}] (для /summary последних сообщений)

const BASE_SYSTEM = `Ты — лаконичный помощник. Пиши по делу, используй Markdown и краткие выводы. Если просят резюме чата — выделяй главные пункты и задачи.`;

const URL_RE = /https?:\/\/\S+/gi;

function getKey(ctx) { return `${ctx.chat.id}:${ctx.from.id}`; }
function pushHistory(ctx, role, content) {
  const k = getKey(ctx);
  if (!sessions.has(k)) sessions.set(k, []);
  const arr = sessions.get(k);
  arr.push({ role, content });
  if (arr.length > MAX_TURNS * 2) arr.shift();
}
function appendChatLog(ctx, role, content) {
  const id = ctx.chat.id;
  const arr = chatLog.get(id) || [];
  arr.push({ role, content });
  if (arr.length > 300) arr.shift();
  chatLog.set(id, arr);
}
function addressedToMe(ctx) {
  if (ctx.chat?.type === 'private') return true;
  const text = ctx.message?.text || '';
  const botUsername = ctx.me?.username || bot.options.username || '';
  if (text.toLowerCase().includes(`@${botUsername.toLowerCase()}`)) return true;
  if (ctx.message?.reply_to_message?.from?.is_bot) return true;
  return false;
}

// ---------- DB ----------
const pool = DATABASE_URL ? new Pool({ connectionString: DATABASE_URL }) : null;

async function dbQuery(q, params = []) {
  if (!pool) return null;
  const client = await pool.connect();
  try { return await client.query(q, params); }
  finally { client.release(); }
}

async function initSchema() {
  if (!pool) return;
  await dbQuery(`CREATE TABLE IF NOT EXISTS user_profile(
    user_id BIGINT PRIMARY KEY,
    system_prompt TEXT DEFAULT ''
  );`);
  await dbQuery(`CREATE TABLE IF NOT EXISTS memory(
    chat_id BIGINT,
    user_id BIGINT,
    ts BIGINT,
    role TEXT,
    content TEXT
  );`);
  await dbQuery(`CREATE TABLE IF NOT EXISTS summaries(
    chat_id BIGINT,
    ts BIGINT,
    summary TEXT
  );`);
}

const sessionRoles = new Map(); // user_id -> 'analyst'|'translator'|'coder'|null

async function storeMessage(ctx, role, content){
  if (!pool) return;
  const ts = Math.floor(Date.now()/1000);
  await dbQuery(
    'INSERT INTO memory(chat_id,user_id,ts,role,content) VALUES ($1,$2,$3,$4,$5)',
    [ctx.chat.id, ctx.from.id, ts, role, content]
  );
}

async function getUserProfile(user_id){
  if (!pool) return '';
  const { rows } = await dbQuery('SELECT system_prompt FROM user_profile WHERE user_id=$1',[user_id]);
  return rows?.[0]?.system_prompt || '';
}

async function setUserProfile(user_id, text){
  if (!pool) return;
  await dbQuery(
    `INSERT INTO user_profile(user_id,system_prompt) VALUES ($1,$2)
     ON CONFLICT (user_id) DO UPDATE SET system_prompt=EXCLUDED.system_prompt`,
    [user_id, text]
  );
}

async function lastSummary(chat_id){
  if (!pool) return '';
  const { rows } = await dbQuery(
    'SELECT summary FROM summaries WHERE chat_id=$1 ORDER BY ts DESC LIMIT 1',
    [chat_id]
  );
  return rows?.[0]?.summary || '';
}

async function addSummary(chat_id, summary){
  if (!pool) return;
  await dbQuery(
    'INSERT INTO summaries(chat_id,ts,summary) VALUES ($1,$2,$3)',
    [chat_id, Math.floor(Date.now()/1000), summary]
  );
}

const ROLES = {
  analyst: 'Роль: аналитик. Делай структурированные выводы, риски и варианты.',
  translator: 'Роль: переводчик. Переводи кратко и точно, указывай язык оригинала.',
  coder: 'Роль: код-ассистент. Пиши код и объясняй шаги максимально ясно.'
};

// ===== утилиты =====
async function fetchAndClean(url) {
  const res = await fetch(url, { headers: { 'User-Agent': 'Mozilla/5.0' } });
  const html = await res.text();
  const $ = load(html);
  $('script,style,noscript').remove();
  let text = $('body').text().replace(/\s+\n/g, '\n').replace(/\n{2,}/g, '\n').trim();
  return text.slice(0, 60000);
}

async function sentiment(text) {
  const r = await openai.chat.completions.create({
    model: LLM_MODEL,
    temperature: 0,
    max_tokens: 4,
    messages: [{ role: 'user', content: `Определи настроение (одно слово: позитив/нейтрально/негативно):\n${text}` }]
  });
  return (r.choices[0].message.content || 'нейтрально').toLowerCase();
}

async function buildMessages(ctx, userText){
  const profile = await getUserProfile(ctx.from.id);
  const roleMode = sessionRoles.get(ctx.from.id);
  const sys = [
    BASE_SYSTEM,
    profile ? `Профиль пользователя: ${profile}` : '',
    roleMode ? ROLES[roleMode] : ''
  ].filter(Boolean).join('\n');

  const msgs = [{ role:'system', content: sys }];
  const sum = await lastSummary(ctx.chat.id);
  if (sum) msgs.push({ role:'system', content:`Краткая память чата: ${sum}` });
  const arr = sessions.get(getKey(ctx)) || [];
  for (const m of arr) msgs.push(m);
  msgs.push({ role:'user', content: userText });
  return msgs;
}

async function handleLLM(ctx, userText) {
  const mood = await sentiment(userText);
  const urls = (userText.match(URL_RE) || []).slice(0, 3);
  let addendum = '';
  if (urls.length) {
    const parts = [];
    for (const u of urls) {
      try {
        const txt = await fetchAndClean(u);
        parts.push(`[${u}] Содержание:\n${txt.slice(0, 3000)}`);
      } catch (e) {
        parts.push(`[${u}] не удалось получить (${e})`);
      }
    }
    addendum = `\n\n---\nВложенные ссылки:\n${parts.join('\n\n')}`;
  }
  const prompt = `Пользователь пишет (тон: ${mood}):\n${userText}${addendum}`;
  const msgs = await buildMessages(ctx, prompt);

  const r = await openai.chat.completions.create({
    model: LLM_MODEL,
    temperature: 0.6,
    max_tokens: 700,
    messages: msgs
  });
  const reply = r.choices[0].message.content || '🤖';
  pushHistory(ctx, 'user', userText);
  pushHistory(ctx, 'assistant', reply);
  appendChatLog(ctx, 'user', userText);
  appendChatLog(ctx, 'assistant', reply);
  await storeMessage(ctx, 'user', userText);
  await storeMessage(ctx, 'assistant', reply);
  await ctx.reply(reply, { disable_web_page_preview: true, parse_mode: 'Markdown' });
}

// ===== команды =====
bot.start(async (ctx) => {
  await ctx.reply(
    'Привет! Я GPT-бот с голосом, ссылками, фото и кратким резюме.\n' +
    'Команды:\n' +
    '/reset — очистить контекст\n' +
    '/setprofile <текст> — персональный стиль ответов для вас\n' +
    '/mode — выбрать роль (аналитик/переводчик/код-ассистент)\n' +
    '/summary — кратко перескажу последние сообщения в чате\n\n' +
    'В группах отвечаю, если меня упомянуть или ответить на моё сообщение.'
  );
});

bot.command('reset', async (ctx) => {
  sessions.delete(getKey(ctx));
  await ctx.reply('Контекст очищён ✅');
});

bot.command('setprofile', async (ctx) => {
  const text = (ctx.message.text || '').split(' ').slice(1).join(' ').trim();
  if (!text) return ctx.reply('Укажите текст профиля: /setprofile ваш_стиль');
  await setUserProfile(ctx.from.id, text);
  await ctx.reply('Профиль сохранён ✅');
});

bot.command('mode', async (ctx) => {
  await ctx.reply('Выберите режим:', Markup.inlineKeyboard([
    [Markup.button.callback('Аналитик', 'mode:analyst')],
    [Markup.button.callback('Переводчик', 'mode:translator')],
    [Markup.button.callback('Код-ассистент', 'mode:coder')],
    [Markup.button.callback('Сбросить', 'mode:clear')]
  ]));
});

bot.action(/^mode:(.+)$/, async (ctx) => {
  const v = ctx.match[1];
  if (v === 'clear') {
    sessionRoles.delete(ctx.from.id);
    await ctx.answerCbQuery('Роль сброшена');
    return ctx.editMessageText('Роль сброшена.');
  }
  sessionRoles.set(ctx.from.id, v);
  await ctx.answerCbQuery(`Роль: ${v}`);
  await ctx.editMessageText(`Роль установлена: ${v}`);
});

bot.command('summary', async (ctx) => {
  const arr = (chatLog.get(ctx.chat.id) || []).slice(-60);
  if (!arr.length) return ctx.reply('История пуста.');
  const text = arr.map(x => `${x.role}: ${x.content}`).join('\n');
  const r = await openai.chat.completions.create({
    model: LLM_MODEL, temperature: 0.2, max_tokens: 380,
    messages: [{ role:'user', content:`Сделай краткое резюме последних сообщений (до 8 пунктов):\n\n${text}` }]
  });
  await addSummary(ctx.chat.id, r.choices[0].message.content || '-');
  await ctx.reply(r.choices[0].message.content || '-', { disable_web_page_preview:true });
});

// ===== текст =====
bot.on('text', async (ctx) => {
  appendChatLog(ctx, 'user', ctx.message.text);
  await storeMessage(ctx, 'user', ctx.message.text);
  if (!addressedToMe(ctx)) return; // в группе — только по упоминанию / reply
  await ctx.sendChatAction('typing');
  await handleLLM(ctx, ctx.message.text);
});

// ===== фото → визуальный разбор =====
bot.on('photo', async (ctx) => {
  if (!addressedToMe(ctx)) return;
  await ctx.sendChatAction('typing');
  const photos = ctx.message.photo;
  const fileId = photos[photos.length - 1].file_id;
  const link = await ctx.telegram.getFileLink(fileId);
  const r = await openai.chat.completions.create({
    model: LLM_MODEL, temperature: 0.4,
    messages: [{
      role: 'user',
      content: [
        { type: 'text', text: 'Опиши картинку и сделай выводы (если есть текст — выпиши кратко).' },
        { type: 'image_url', image_url: { url: link.href } }
      ]
    }]
  });
  const reply = r.choices[0].message.content || '🖼️';
  pushHistory(ctx, 'user', '(image)');
  pushHistory(ctx, 'assistant', reply);
  appendChatLog(ctx, 'user', '(image)');
  appendChatLog(ctx, 'assistant', reply);
  await storeMessage(ctx, 'user', '(image)');
  await storeMessage(ctx, 'assistant', reply);
  await ctx.reply(reply, { disable_web_page_preview: true });
});

// ===== голос =====
bot.on('voice', async (ctx) => {
  if (!addressedToMe(ctx)) return;
  await ctx.sendChatAction('typing');
  const link = await ctx.telegram.getFileLink(ctx.message.voice.file_id);
  const buf = Buffer.from(await (await fetch(link.href)).arrayBuffer());
  const file = await toFile(buf, 'voice.ogg', { type: 'audio/ogg' });
  const tr = await openai.audio.transcriptions.create({ model: ASR_MODEL, file });
  await storeMessage(ctx, 'user', '(voice)');
  await handleLLM(ctx, tr.text || '(пустая расшифровка)');
});

// ===== «кружки» (video_note) =====
bot.on('video_note', async (ctx) => {
  if (!addressedToMe(ctx)) return;
  await ctx.sendChatAction('typing');
  const link = await ctx.telegram.getFileLink(ctx.message.video_note.file_id);
  const buf = Buffer.from(await (await fetch(link.href)).arrayBuffer());
  const file = await toFile(buf, 'circle.mp4', { type: 'video/mp4' });
  const tr = await openai.audio.transcriptions.create({ model: ASR_MODEL, file });
  await storeMessage(ctx, 'user', '(video_note)');
  await handleLLM(ctx, tr.text || '(пустая расшифровка)');
});

// ===== WEBHOOK & STARTUP =====
app.use(bot.webhookCallback(WEBHOOK_SECRET_PATH));
app.get('/', (_, res) => res.send('OK'));

app.listen(PORT, async () => {
  console.log('HTTP server listening on port', PORT);
  await initSchema(); // создаём таблицы, если есть DATABASE_URL
  const url = `${PUBLIC_URL}${WEBHOOK_SECRET_PATH}`;
  await bot.telegram.setWebhook(url);
  const me = await bot.telegram.getMe();
  bot.options.username = me.username;
  console.log(`Bot @${me.username} webhook set to ${url}`);
});

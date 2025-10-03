import 'dotenv/config';
import express from 'express';
import { Telegraf, Markup } from 'telegraf';
import { OpenAI, toFile } from 'openai';
import { load } from 'cheerio';;

// ===== ENV =====
const TG_TOKEN = process.env.TELEGRAM_BOT_TOKEN;
const OPENAI_KEY = process.env.OPENAI_API_KEY;
const LLM_MODEL = process.env.OPENAI_MODEL || 'gpt-4o-mini';
const ASR_MODEL = process.env.OPENAI_TRANSCRIBE || 'whisper-1';
const PUBLIC_URL = process.env.PUBLIC_URL;
const WEBHOOK_SECRET_PATH = process.env.WEBHOOK_SECRET_PATH || '/bot';
const PORT = Number(process.env.PORT || 8080);

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
const chatLog = new Map(); // ключ: chat_id -> [{role,content}] (для /summary последних сообщений)

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
  if (arr.length > 200) arr.shift();
  chatLog.set(id, arr);
}
function addressedToMe(ctx) {
  if (ctx.chat?.type === 'private') return true;
  const text = ctx.message?.text || '';
  const botUsername = ctx.me?.username || '';
  if (text.toLowerCase().includes(`@${botUsername.toLowerCase()}`)) return true;
  if (ctx.message?.reply_to_message?.from?.is_bot) return true;
  return false;
}

// ===== утилиты =====
async function fetchAndClean(url) {
  const res = await fetch(url, { headers: { 'User-Agent': 'Mozilla/5.0' }, timeout: 12000 });
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
async function buildMessages(ctx, userText) {
  const msgs = [{ role: 'system', content: BASE_SYSTEM }];
  const arr = sessions.get(getKey(ctx)) || [];
  for (const m of arr) msgs.push(m);
  msgs.push({ role: 'user', content: userText });
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
  await ctx.reply(reply, { disable_web_page_preview: true, parse_mode: 'Markdown' });
}

// ===== команды =====
bot.start(async (ctx) => {
  await ctx.reply(
    'Привет! Я GPT-бот с голосом, ссылками, фото и кратким резюме.\n' +
    'Команды:\n' +
    '/reset — очистить контекст\n' +
    '/summary — кратко перескажу последние сообщения в чате\n\n' +
    'В группах отвечаю, если меня упомянуть или ответить на моё сообщение.'
  );
});
bot.command('reset', async (ctx) => {
  sessions.delete(getKey(ctx));
  await ctx.reply('Контекст очищён ✅');
});
bot.command('summary', async (ctx) => {
  const arr = (chatLog.get(ctx.chat.id) || []).slice(-50);
  if (!arr.length) return ctx.reply('История пуста.');
  const text = arr.map(x => `${x.role}: ${x.content}`).join('\n');
  const r = await openai.chat.completions.create({
    model: LLM_MODEL, temperature: 0.2, max_tokens: 400,
    messages: [{ role: 'user', content: `Сделай краткое резюме последних сообщений (до 8 пунктов):\n\n${text}` }]
  });
  await ctx.reply(r.choices[0].message.content || '—', { disable_web_page_preview: true });
});

// ===== текст =====
bot.on('text', async (ctx) => {
  appendChatLog(ctx, 'user', ctx.message.text);
  if (!addressedToMe(ctx)) return;      // в группе — только по упоминанию / reply
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
  await handleLLM(ctx, tr.text || '(пустая расшифровка)');
});

// ===== WEBHOOK =====
app.use(bot.webhookCallback(WEBHOOK_SECRET_PATH));
app.get('/', (_, res) => res.send('OK'));

app.listen(PORT, async () => {
  const url = `${PUBLIC_URL}${WEBHOOK_SECRET_PATH}`;
  await bot.telegram.setWebhook(url);
  const me = await bot.telegram.getMe();
  bot.options.username = me.username;
  console.log(`Bot @${me.username} webhook set to ${url}`);
});

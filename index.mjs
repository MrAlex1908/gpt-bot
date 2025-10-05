// index.mjs
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

if (!TG_TOKEN) throw new Error('TELEGRAM_BOT_TOKEN is required');
if (!OPENAI_KEY) throw new Error('OPENAI_API_KEY is required');
if (!PUBLIC_URL) throw new Error('PUBLIC_URL is required for webhooks');

const openai = new OpenAI({ apiKey: OPENAI_KEY });
const bot = new Telegraf(TG_TOKEN);

const app = express();
app.use(express.json());

// ===== оперативная память (RAM) =====
const MAX_TURNS = 8;
const sessions = new Map(); // `${chat_id}:${user_id}` -> [{role,content}]
const chatLog  = new Map(); // chat_id -> [{role,content}] (для /summary)

const BASE_SYSTEM = `Ты — лаконичный помощник. Пиши по делу, используй Markdown и краткие выводы. Если просят резюме чата — выделяй главные пункты и задачи.`;

function getKey(ctx){ return `${ctx.chat.id}:${ctx.from.id}`; }
function pushHistory(ctx, role, content){
  const k = getKey(ctx);
  if (!sessions.has(k)) sessions.set(k, []);
  const arr = sessions.get(k);
  arr.push({ role, content });
  if (arr.length > MAX_TURNS*2) arr.shift();
}
function appendChatLog(ctx, role, content){
  const id = ctx.chat.id;
  const arr = chatLog.get(id) || [];
  arr.push({ role, content });
  if (arr.length > 300) arr.shift();
  chatLog.set(id, arr);
}
function addressedToMe(ctx){
  if (ctx.chat?.type === 'private') return true;
  const text = ctx.message?.text || '';
  const botUsername = ctx.me?.username || bot.options.username || '';
  if (botUsername && text.toLowerCase().includes(`@${botUsername.toLowerCase()}`)) return true;
  if (ctx.message?.reply_to_message?.from?.is_bot) return true;
  return false;
}

// ===== helper: разрешаем команды только в ЛС =====
function ensurePrivate(ctx) {
  if (ctx.chat?.type !== 'private') {
    ctx.reply('Эта команда работает только в личке. Напишите мне в ЛС.');
    return false;
  }
  return true;
}

// ===== helper: проверка, что бот админ в канале =====
async function botIsAdmin(chatId, tg) {
  try {
    const admins = await tg.getChatAdministrators(chatId);
    const me = await tg.getMe();
    return admins.some(a => a.user.id === me.id);
  } catch (_) {
    return false;
  }
}

// ===== /linkchannel @канал_или_id =====
bot.command('linkchannel', async (ctx) => {
  if (!ensurePrivate(ctx)) return;
  const arg = (ctx.message.text || '').split(' ').slice(1).join(' ').trim();
  if (!arg) return ctx.reply('Формат: /linkchannel @username_канала или числовой ID');

  try {
    const chat = await ctx.telegram.getChat(arg);
    if (chat.type !== 'channel') {
      return ctx.reply('Это не канал. Укажите @username или ID канала.');
    }
    const isAdmin = await botIsAdmin(chat.id, ctx.telegram);
    if (!isAdmin) {
      return ctx.reply('Я не админ в этом канале. Добавьте меня админом и повторите.');
    }

    await upsertChat(chat, { can_post: true });
    await upsertUser(ctx.from);
    await linkUserChannel(ctx.from.id, chat.id);

    await ctx.reply(`Канал привязан: ${chat.title || chat.username || chat.id}\nТеперь можно: /post Текст поста`);
  } catch (e) {
    await ctx.reply('Не удалось найти канал или нет прав. Проверьте @username/ID и мои права администратора.');
  }
});

// ===== /mychannels =====
bot.command('mychannels', async (ctx) => {
  if (!ensurePrivate(ctx)) return;
  const rows = await getUserChannels(ctx.from.id);
  if (!rows.length) return ctx.reply('Нет привязанных каналов. Используйте /linkchannel @канал');
  const list = rows.map((r, i) =>
    `${i+1}. ${r.title || r.username || r.chat_id}  ${r.bot_can_post ? '✅ постинг' : '⛔️'}`
  ).join('\n');
  await ctx.reply(`Ваши каналы:\n${list}`);
});

// ===== /unlinkchannel @канал_или_id =====
bot.command('unlinkchannel', async (ctx) => {
  if (!ensurePrivate(ctx)) return;
  const arg = (ctx.message.text || '').split(' ').slice(1).join(' ').trim();
  if (!arg) return ctx.reply('Формат: /unlinkchannel @username_канала или ID');

  try {
    const chat = await ctx.telegram.getChat(arg);
    await dbQuery(`DELETE FROM user_channels WHERE user_id=$1 AND chat_id=$2`, [ctx.from.id, chat.id]);
    await ctx.reply(`Канал отвязан: ${chat.title || chat.username || chat.id}`);
  } catch (e) {
    await ctx.reply('Не нашёл канал. Проверьте аргумент.');
  }
});

// ===== /digest @канал [N] =====
bot.command('digest', async (ctx) => {
  if (!ensurePrivate(ctx)) return;

  const parts = (ctx.message.text || '').split(/\s+/).slice(1);
  if (!parts.length) return ctx.reply('Формат: /digest @канал 50  (по умолчанию N=50)');

  const target = parts[0];
  const N = Math.max(5, Math.min(200, parseInt(parts[1] || '50', 10) || 50));

  try {
    const chat = await ctx.telegram.getChat(target);
    if (chat.type !== 'channel') return ctx.reply('Укажите именно канал: @username или ID.');

    const { rows } = await dbQuery(
      `SELECT content, media_type FROM messages
         WHERE chat_id=$1 AND role='channel' AND deleted=FALSE
         ORDER BY ts DESC LIMIT $2`,
      [chat.id, N]
    );

    if (!rows?.length) {
      return ctx.reply('В базе нет постов этого канала (добавьте бота в канал, чтобы он видел посты).');
    }

    const plain = rows
      .map((r, i) => `${i+1}) [${r.media_type}] ${r.content || '(media)'}`)
      .join('\n');

    const prompt = `Сделай краткий дайджест канала. До 10 пунктов, по делу, без воды.
Тексты постов (новые сверху):
${plain}`;

    const r = await openai.chat.completions.create({
      model: LLM_MODEL,
      temperature: 0.3,
      max_tokens: 500,
      messages: [{ role: 'user', content: prompt }]
    });

    await ctx.reply(r.choices[0].message.content || '—', { disable_web_page_preview: true });
  } catch (e) {
    await ctx.reply('Ошибка. Проверьте @канал и что бот добавлен в канал.');
  }
});

// ---------- DB (v2) ----------
const pool = process.env.DATABASE_URL ? new Pool({ connectionString: process.env.DATABASE_URL }) : null;

async function dbQuery(q, params = []){
  if (!pool) return null;
  const c = await pool.connect();
  try { return await c.query(q, params); }
  finally { c.release(); }
}

export async function initSchema(){
  if (!pool) return;
  await dbQuery(`CREATE EXTENSION IF NOT EXISTS "uuid-ossp";`);

  await dbQuery(`
    CREATE TABLE IF NOT EXISTS users(
      user_id        BIGINT PRIMARY KEY,
      username       TEXT,
      first_name     TEXT,
      last_name      TEXT,
      language_code  TEXT,
      system_prompt  TEXT DEFAULT '',
      settings_json  JSONB DEFAULT '{}'::jsonb,
      created_at     TIMESTAMPTZ DEFAULT NOW(),
      updated_at     TIMESTAMPTZ DEFAULT NOW()
    );`);

  await dbQuery(`
    CREATE TABLE IF NOT EXISTS chats(
      chat_id        BIGINT PRIMARY KEY,
      type           TEXT,
      title          TEXT,
      username       TEXT,
      bot_can_post   BOOLEAN DEFAULT FALSE,
      bot_can_react  BOOLEAN DEFAULT TRUE,
      created_at     TIMESTAMPTZ DEFAULT NOW(),
      updated_at     TIMESTAMPTZ DEFAULT NOW()
    );`);

  await dbQuery(`
    CREATE TABLE IF NOT EXISTS user_channels(
      user_id   BIGINT REFERENCES users(user_id) ON DELETE CASCADE,
      chat_id   BIGINT REFERENCES chats(chat_id) ON DELETE CASCADE,
      created_at TIMESTAMPTZ DEFAULT NOW(),
      PRIMARY KEY(user_id, chat_id)
    );`);

  await dbQuery(`
    CREATE TABLE IF NOT EXISTS messages(
      chat_id             BIGINT NOT NULL,
      message_id          BIGINT NOT NULL,
      user_id             BIGINT,
      role                TEXT,   -- user | assistant | channel | system
      content             TEXT,
      media_type          TEXT,   -- text | photo | voice | video_note | ...
      media_url           TEXT,
      reply_to_message_id BIGINT,
      thread_id           BIGINT,
      ts                  BIGINT, -- unix seconds
      edited              BOOLEAN DEFAULT FALSE,
      deleted             BOOLEAN DEFAULT FALSE,
      extra               JSONB DEFAULT '{}'::jsonb,
      PRIMARY KEY(chat_id, message_id)
    );`);

  // ✅ FIXED: без COALESCE в PK
  await dbQuery(`
    CREATE TABLE IF NOT EXISTS reactions(
      chat_id    BIGINT NOT NULL,
      message_id BIGINT NOT NULL,
      emoji      TEXT   NOT NULL,
      user_id    BIGINT NOT NULL DEFAULT 0,
      ts         BIGINT,
      PRIMARY KEY(chat_id, message_id, emoji, user_id)
    );`);

  await dbQuery(`
    CREATE TABLE IF NOT EXISTS posts_log(
      id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
      chat_id    BIGINT NOT NULL,
      message_id BIGINT,
      user_id    BIGINT,
      text       TEXT,
      ts         BIGINT,
      status     TEXT,
      error      TEXT
    );`);

  await dbQuery(`
    CREATE TABLE IF NOT EXISTS summaries(
      chat_id BIGINT,
      ts      BIGINT,
      summary TEXT,
      period  TEXT DEFAULT 'last',
      PRIMARY KEY(chat_id, ts)
    );`);

  await dbQuery(`CREATE INDEX IF NOT EXISTS idx_messages_chat_ts ON messages(chat_id, ts DESC);`);
  await dbQuery(`CREATE INDEX IF NOT EXISTS idx_messages_user_ts ON messages(user_id, ts DESC);`);
  await dbQuery(`CREATE INDEX IF NOT EXISTS idx_reactions_chat_msg ON reactions(chat_id, message_id);`);
}

// апсерты/утилиты БД
export async function upsertUser(from = {}){
  if (!pool || !from) return;
  await dbQuery(
    `INSERT INTO users(user_id, username, first_name, last_name, language_code)
     VALUES ($1,$2,$3,$4,$5)
     ON CONFLICT(user_id) DO UPDATE SET
       username=EXCLUDED.username,
       first_name=EXCLUDED.first_name,
       last_name=EXCLUDED.last_name,
       language_code=EXCLUDED.language_code,
       updated_at=NOW();`,
    [from.id, from.username || null, from.first_name || null, from.last_name || null, from.language_code || null]
  );
}
export async function upsertChat(chat = {}, rights = {}){
  if (!pool || !chat) return;
  await dbQuery(
    `INSERT INTO chats(chat_id, type, title, username, bot_can_post, bot_can_react)
     VALUES ($1,$2,$3,$4,$5,$6)
     ON CONFLICT(chat_id) DO UPDATE SET
       type=EXCLUDED.type,
       title=EXCLUDED.title,
       username=EXCLUDED.username,
       bot_can_post=EXCLUDED.bot_can_post,
       bot_can_react=EXCLUDED.bot_can_react,
       updated_at=NOW();`,
    [chat.id, chat.type || null, chat.title || null, chat.username || null, rights?.can_post ?? false, rights?.can_react ?? true]
  );
}
export async function linkUserChannel(user_id, chat_id){
  if (!pool) return;
  await dbQuery(
    `INSERT INTO user_channels(user_id, chat_id) VALUES ($1,$2)
     ON CONFLICT(user_id, chat_id) DO NOTHING;`,
    [user_id, chat_id]
  );
}
export async function getUserChannels(user_id){
  if (!pool) return [];
  const { rows } = await dbQuery(
    `SELECT uc.chat_id, c.title, c.username, c.type, c.bot_can_post
       FROM user_channels uc LEFT JOIN chats c ON c.chat_id = uc.chat_id
      WHERE uc.user_id=$1 ORDER BY c.title NULLS LAST;`,
    [user_id]
  );
  return rows || [];
}
export async function storeMessageV2({
  chat_id, message_id, user_id = null,
  role = 'user', content = '', media_type = 'text',
  media_url = null, reply_to_message_id = null,
  thread_id = null, ts = Math.floor(Date.now()/1000), extra = {}
}){
  if (!pool) return;
  await dbQuery(
    `INSERT INTO messages(chat_id, message_id, user_id, role, content, media_type, media_url, reply_to_message_id, thread_id, ts, extra)
     VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
     ON CONFLICT (chat_id, message_id)
     DO UPDATE SET content=EXCLUDED.content,
                   media_type=EXCLUDED.media_type,
                   media_url=EXCLUDED.media_url,
                   reply_to_message_id=EXCLUDED.reply_to_message_id,
                   thread_id=EXCLUDED.thread_id,
                   ts=EXCLUDED.ts,
                   extra=EXCLUDED.extra;`,
    [chat_id, message_id, user_id, role, content, media_type, media_url, reply_to_message_id, thread_id, ts, extra]
  );
}
export async function storeReaction({
  chat_id, message_id, emoji = '👍', user_id = null, ts = Math.floor(Date.now()/1000)
}){
  if (!pool) return;
  await dbQuery(
    `INSERT INTO reactions(chat_id, message_id, emoji, user_id, ts)
     VALUES ($1,$2,$3,$4,$5)
     ON CONFLICT(chat_id, message_id, emoji, user_id) DO NOTHING;`,
    [chat_id, message_id, emoji, (user_id ?? 0), ts]
  );
}
export async function logPost({ chat_id, message_id = null, user_id = null, text = '', status = 'sent', error = null }){
  if (!pool) return;
  await dbQuery(
    `INSERT INTO posts_log(chat_id, message_id, user_id, text, ts, status, error)
     VALUES ($1,$2,$3,$4,$5,$6,$7);`,
    [chat_id, message_id, user_id, text, Math.floor(Date.now()/1000), status, error]
  );
}
export async function addSummary(chat_id, summary, period='last'){
  if (!pool) return;
  await dbQuery(
    `INSERT INTO summaries(chat_id, ts, summary, period) VALUES ($1,$2,$3,$4);`,
    [chat_id, Math.floor(Date.now()/1000), summary, period]
  );
}
export async function lastSummary(chat_id){
  if (!pool) return '';
  const { rows } = await dbQuery(
    `SELECT summary FROM summaries WHERE chat_id=$1 ORDER BY ts DESC LIMIT 1;`,
    [chat_id]
  );
  return rows?.[0]?.summary || '';
}

// ---- роли/профиль пользователя ----
const sessionRoles = new Map(); // user_id -> 'analyst'|'translator'|'coder'|null
const ROLES = {
  analyst: 'Роль: аналитик. Делай структурированные выводы, риски и варианты.',
  translator: 'Роль: переводчик. Переводи кратко и точно, указывай язык оригинала.',
  coder: 'Роль: код-ассистент. Пиши код и объясняй шаги максимально ясно.'
};
async function getUserProfile(user_id){
  const { rows } = await dbQuery('SELECT system_prompt FROM users WHERE user_id=$1',[user_id]);
  return rows?.[0]?.system_prompt || '';
}
async function setUserProfile(user_id, text){
  await dbQuery(
    `INSERT INTO users(user_id, system_prompt) VALUES ($1,$2)
     ON CONFLICT (user_id) DO UPDATE SET system_prompt=EXCLUDED.system_prompt, updated_at=NOW()`,
    [user_id, text]
  );
}

// ===== вспомогательные утилиты =====
async function fetchAndClean(url){
  const res = await fetch(url, { headers: { 'User-Agent': 'Mozilla/5.0' } });
  const html = await res.text();
  const $ = load(html);
  $('script,style,noscript').remove();
  let text = $('body').text().replace(/\s+\n/g, '\n').replace(/\n{2,}/g, '\n').trim();
  return text.slice(0, 60000);
}
async function sentiment(text){
  const r = await openai.chat.completions.create({
    model: LLM_MODEL, temperature: 0, max_tokens: 4,
    messages: [{ role:'user', content:`Определи настроение (одно слово: позитив/нейтрально/негативно):\n${text}` }]
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

// ===== команды =====
bot.start(async (ctx)=>{
  await ctx.reply(
    'Привет! Я GPT-бот с голосом, ссылками, фото и кратким резюме.\n' +
    'Команды:\n' +
    '/reset — очистить контекст\n' +
    '/setprofile <текст> — персональный стиль ответов\n' +
    '/mode — выбрать роль (аналитик/переводчик/код-ассистент)\n' +
    '/summary — кратко перескажу последние сообщения\n\n' +
    'В группах отвечаю по упоминанию или reply.'
  );
});
bot.command('reset', async (ctx)=>{
  sessions.delete(getKey(ctx));
  await ctx.reply('Контекст очищён ✅');
});
bot.command('setprofile', async (ctx)=>{
  const text = (ctx.message.text || '').split(' ').slice(1).join(' ').trim();
  if (!text) return ctx.reply('Укажите текст профиля: /setprofile ваш_стиль');
  await setUserProfile(ctx.from.id, text);
  await ctx.reply('Профиль сохранён ✅');
});
bot.command('mode', async (ctx)=>{
  await ctx.reply('Выберите режим:', Markup.inlineKeyboard([
    [Markup.button.callback('Аналитик','mode:analyst')],
    [Markup.button.callback('Переводчик','mode:translator')],
    [Markup.button.callback('Код-ассистент','mode:coder')],
    [Markup.button.callback('Сбросить','mode:clear')]
  ]));
});
bot.action(/^mode:(.+)$/, async (ctx)=>{
  const v = ctx.match[1];
  if (v === 'clear'){
    sessionRoles.delete(ctx.from.id);
    await ctx.answerCbQuery('Роль сброшена');
    return ctx.editMessageText('Роль сброшена.');
  }
  sessionRoles.set(ctx.from.id, v);
  await ctx.answerCbQuery(`Роль: ${v}`);
  await ctx.editMessageText(`Роль установлена: ${v}`);
});
bot.command('summary', async (ctx)=>{
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

// ===== Handlers & helpers (Block B, unified) =====
async function makeLLMReply(ctx, userText){
  const mood = await sentiment(userText);
  const URL_RE = /https?:\/\/\S+/gi;
  const urls = (userText.match(URL_RE) || []).slice(0, 3);

  let addendum = '';
  if (urls.length){
    const parts = [];
    for (const u of urls){
      try {
        const txt = await fetchAndClean(u);
        parts.push(`[${u}] Содержание:\n${txt.slice(0, 3000)}`);
      } catch(e){
        parts.push(`[${u}] не удалось получить (${e})`);
      }
    }
    addendum = `\n\n---\nВложенные ссылки:\n${parts.join('\n\n')}`;
  }

  const msgs = await buildMessages(ctx, `Пользователь пишет (тон: ${mood}):\n${userText}${addendum}`);
  const r = await openai.chat.completions.create({
    model: LLM_MODEL, temperature: 0.6, max_tokens: 700, messages: msgs
  });
  return r.choices[0].message.content || '🤖';
}

async function replyAndStore(ctx, text, extra={}){
  const m = await ctx.reply(text, { disable_web_page_preview:true, parse_mode:'Markdown', ...extra });
  await storeMessageV2({ chat_id: ctx.chat.id, message_id: m.message_id, role:'assistant', content:text, media_type:'text' });
  pushHistory(ctx, 'assistant', text);
  appendChatLog(ctx, 'assistant', text);
  return m;
}

async function sendPostToChannel(chatId, text, requestedByUserId){
  try{
    const msg = await bot.telegram.sendMessage(chatId, text, { disable_web_page_preview:false, parse_mode:'Markdown' });
    await logPost({ chat_id: chatId, message_id: msg.message_id, user_id: requestedByUserId, text, status:'sent' });
    await storeMessageV2({ chat_id: chatId, message_id: msg.message_id, role:'assistant', content:text, media_type:'text' });
    return msg;
  }catch(e){
    await logPost({ chat_id: chatId, message_id:null, user_id: requestedByUserId, text, status:'failed', error:String(e) });
    throw e;
  }
}

async function reactToMessage(chat_id, message_id, emojis=['👍'], fromUserId=null){
  const reaction = emojis.map(e => ({ type:'emoji', emoji:e }));
  await bot.telegram.callApi('setMessageReaction', { chat_id, message_id, reaction });
  for (const e of emojis) await storeReaction({ chat_id, message_id, emoji:e, user_id:fromUserId });
}

// ---------- TEXT ----------
bot.on('text', async (ctx)=>{
  await upsertUser(ctx.from);
  await upsertChat(ctx.chat);

  const m = ctx.message;

  await storeMessageV2({
    chat_id: ctx.chat.id,
    message_id: m.message_id,
    user_id: ctx.from.id,
    role: 'user',
    content: m.text,
    media_type: 'text',
    reply_to_message_id: m.reply_to_message?.message_id || null,
    thread_id: m.message_thread_id || null
  });
  pushHistory(ctx, 'user', m.text);
  appendChatLog(ctx, 'user', m.text);

  if (!addressedToMe(ctx)) return;

  await ctx.sendChatAction('typing');
  const replyText = await makeLLMReply(ctx, m.text);
  await replyAndStore(ctx, replyText);
});

// ---------- PHOTO ----------
bot.on('photo', async (ctx)=>{
  await upsertUser(ctx.from);
  await upsertChat(ctx.chat);

  const m = ctx.message;
  const fileId = m.photo.at(-1).file_id;
  const link = await ctx.telegram.getFileLink(fileId);

  await storeMessageV2({
    chat_id: ctx.chat.id,
    message_id: m.message_id,
    user_id: ctx.from.id,
    role: 'user',
    content: m.caption || '',
    media_type: 'photo',
    media_url: link.href,
    reply_to_message_id: m.reply_to_message?.message_id || null,
    thread_id: m.message_thread_id || null
  });
  pushHistory(ctx, 'user', '(image)');
  appendChatLog(ctx, 'user', '(image)');

  if (!addressedToMe(ctx)) return;

  await ctx.sendChatAction('typing');
  const r = await openai.chat.completions.create({
    model: LLM_MODEL, temperature:0.4,
    messages: [{
      role:'user',
      content:[
        { type:'text', text:'Опиши картинку и сделай выводы (если есть текст — выпиши кратко).' },
        { type:'image_url', image_url:{ url: link.href } }
      ]
    }]
  });
  const replyText = r.choices[0].message.content || '🖼️';
  await replyAndStore(ctx, replyText);
});

// ---------- VOICE ----------
bot.on('voice', async (ctx)=>{
  await upsertUser(ctx.from);
  await upsertChat(ctx.chat);

  const m = ctx.message;
  const link = await ctx.telegram.getFileLink(m.voice.file_id);

  await storeMessageV2({
    chat_id: ctx.chat.id,
    message_id: m.message_id,
    user_id: ctx.from.id,
    role: 'user',
    content: '(voice)',
    media_type: 'voice',
    media_url: link.href,
    reply_to_message_id: m.reply_to_message?.message_id || null,
    thread_id: m.message_thread_id || null
  });
  pushHistory(ctx, 'user', '(voice)');
  appendChatLog(ctx, 'user', '(voice)');

  if (!addressedToMe(ctx)) return;

  await ctx.sendChatAction('typing');
  const buf = Buffer.from(await (await fetch(link.href)).arrayBuffer());
  const file = await toFile(buf, 'voice.ogg', { type:'audio/ogg' });
  const tr = await openai.audio.transcriptions.create({ model: ASR_MODEL, file });
  const replyText = await makeLLMReply(ctx, tr.text || '(пустая расшифровка)');
  await replyAndStore(ctx, replyText);
});

// ---------- VIDEO NOTE ----------
bot.on('video_note', async (ctx)=>{
  await upsertUser(ctx.from);
  await upsertChat(ctx.chat);

  const m = ctx.message;
  const link = await ctx.telegram.getFileLink(m.video_note.file_id);

  await storeMessageV2({
    chat_id: ctx.chat.id,
    message_id: m.message_id,
    user_id: ctx.from.id,
    role: 'user',
    content: '(video_note)',
    media_type: 'video_note',
    media_url: link.href,
    reply_to_message_id: m.reply_to_message?.message_id || null,
    thread_id: m.message_thread_id || null
  });
  pushHistory(ctx, 'user', '(video_note)');
  appendChatLog(ctx, 'user', '(video_note)');

  if (!addressedToMe(ctx)) return;

  await ctx.sendChatAction('typing');
  const buf = Buffer.from(await (await fetch(link.href)).arrayBuffer());
  const file = await toFile(buf, 'circle.mp4', { type:'video/mp4' });
  const tr = await openai.audio.transcriptions.create({ model: ASR_MODEL, file });
  const replyText = await makeLLMReply(ctx, tr.text || '(пустая расшифровка)');
  await replyAndStore(ctx, replyText);
});

// ---------- CHANNEL POSTS ----------
bot.on('channel_post', async (ctx)=>{
  await upsertChat(ctx.chat, { can_post:true });

  const p = ctx.channelPost;
  const text = p.text || p.caption || '';
  let mediaUrl = null;
  let mediaType = 'text';

  if (p.photo?.length){
    mediaType = 'photo';
    mediaUrl = (await ctx.telegram.getFileLink(p.photo.at(-1).file_id)).href;
  }

  await storeMessageV2({
    chat_id: ctx.chat.id,
    message_id: p.message_id,
    role: 'channel',
    content: text,
    media_type: mediaType,
    media_url: mediaUrl,
    thread_id: p.message_thread_id || null,
    ts: p.date
  });
});

// ---------- OPTIONAL: react & post ----------
bot.command('post', async (ctx)=>{
  const raw = (ctx.message.text || '').slice(5).trim();
  if (!raw) return ctx.reply('Формат: /post @канал Текст поста\nили: /post Текст (если привязан один канал)');
  const parts = raw.split(/\s+/);
  let chan = null, text = raw;
  if (parts[0].startsWith('@') || /^\-?\d+$/.test(parts[0])) { chan = parts[0]; text = raw.slice(parts[0].length).trim(); }

  let chatId = null;
  if (chan){
    try { const chat = await ctx.telegram.getChat(chan); chatId = chat.id; } catch {}
  }else{
    const list = await getUserChannels(ctx.from.id);
    if (list.length === 1) chatId = list[0].chat_id;
    else return ctx.reply('Несколько каналов привязано. Укажите: /post @канал Текст поста');
  }
  if (!chatId) return ctx.reply('Не нашёл канал. Проверьте /mychannels или /linkchannel.');

  try { await sendPostToChannel(chatId, text, ctx.from.id); await ctx.reply('Опубликовано ✅'); }
  catch { await ctx.reply('Не удалось опубликовать. Дайте боту права Админа (Post Messages).'); }
});

bot.command('react', async (ctx)=>{
  const emo = (ctx.message.text || '').split(' ')[1] || '👍';
  const tgt = ctx.message?.reply_to_message;
  if (!tgt) return ctx.reply('Ответьте на сообщение командой: /react 👍');
  try { await reactToMessage(ctx.chat.id, tgt.message_id, [emo], ctx.from.id); } catch {}
});

// ===== WEBHOOK & STARTUP =====
app.use(bot.webhookCallback(WEBHOOK_SECRET_PATH));
app.get('/', (_, res) => res.send('OK'));

app.listen(PORT, async ()=>{
  console.log('HTTP server listening on port', PORT);
  await initSchema(); // создаст таблицы, если есть DATABASE_URL
  const url = `${PUBLIC_URL}${WEBHOOK_SECRET_PATH}`;
  await bot.telegram.setWebhook(url);
  const me = await bot.telegram.getMe();
  bot.options.username = me.username;
  console.log(`Bot @${me.username} webhook set to ${url}`);
});

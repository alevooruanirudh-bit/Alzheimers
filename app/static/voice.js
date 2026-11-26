/*
 * Voice help for people who cannot easily read or write.
 *
 * - Questions are read aloud with the browser's speech synthesis (works in most browsers).
 * - Answers are spoken and recognised with the browser's speech recognition
 *   (Chrome / Edge; needs a microphone and an internet connection). The app then
 *   says back what it understood, so the person can correct it.
 *
 * Any element with a data-voice attribute is a question:
 *   data-voice = yesno | help | select | number | text | height | weight |
 *                words3 | numbers5 | buttons | command | gate
 *   data-name  = the form field(s) it fills (comma-separated for words3 / numbers5)
 *   data-say   = the text read aloud
 *
 * A test hook (window.__voiceTest = {queue: [...]}) replaces the microphone with
 * scripted answers and makes speech instant, so the flow can be tested automatically.
 */
(() => {
  const form = document.querySelector('form[data-voice-form]');
  if (!form) return;
  const Rec = window.SpeechRecognition || window.webkitSpeechRecognition;
  const test = () => window.__voiceTest;
  const canSpeak = 'speechSynthesis' in window;
  const canListen = !!Rec || !!test();
  let guideOn = false, runId = 0, currentRec = null, guidedClick = false;
  // clicks made by the guide itself must not re-trigger the guide
  function gclick(el) { guidedClick = true; try { el.click(); } finally { guidedClick = false; } }

  // ------------------------------------------------------------------ speech output
  function pickVoice() {
    const voices = canSpeak ? speechSynthesis.getVoices() : [];
    return voices.find(v => v.lang === 'en-IN') || voices.find(v => v.lang.startsWith('en-GB'))
      || voices.find(v => v.lang.startsWith('en')) || null;
  }
  function say(text) {
    if (!text) return Promise.resolve();
    if (test()) { (test().spoken = test().spoken || []).push(text); return Promise.resolve(); }
    if (!canSpeak) return Promise.resolve();
    return new Promise(resolve => {
      speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      const v = pickVoice();
      if (v) u.voice = v;
      u.lang = v ? v.lang : 'en-IN';
      u.rate = 0.92;
      const done = () => { clearTimeout(t); resolve(); };
      const t = setTimeout(done, 2500 + text.length * 90);  // safety net if 'end' never fires
      u.onend = done; u.onerror = done;
      speechSynthesis.speak(u);
    });
  }
  let audioCtx = null;
  function beep() {  // tells the person "speak now"
    if (test()) return Promise.resolve();
    try {
      audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
      if (audioCtx.state === 'suspended') audioCtx.resume();
      const o = audioCtx.createOscillator(), g = audioCtx.createGain();
      o.frequency.value = 880; g.gain.value = 0.08;
      o.connect(g); g.connect(audioCtx.destination); o.start(); o.stop(audioCtx.currentTime + 0.12);
    } catch (e) { /* no audio: carry on */ }
    return new Promise(r => setTimeout(r, 200));
  }

  // ------------------------------------------------------------------ speech input
  // Listening, tuned for one-word answers ("yes", "five"). A word said inside a sentence is understood, but a
  // single word on its own can be lost, because the start of the sound is dropped while the microphone starts up
  // (Bluetooth earphones switch into "headset mode" at that moment) or while echo cancellation settles after the
  // beep. So:
  //  - the microphone is opened once and kept open while voice help is being used ("warm"), so later questions
  //    do not wait for the device;
  //  - the recogniser is started first; the beep plays only after it has been receiving sound for a moment;
  //  - continuous mode, stopped by us about a second after the person pauses;
  //  - every guess is returned (best first), or the in-progress guess if no final one came;
  //  - a retry uses the general English model as well, which sometimes catches short words the Indian one misses.
  // Each attempt is written to a local troubleshooting log (database/voice_debug.log) with its timings.
  let lastHeard = '', micStream = null, micIdle = null;
  async function warmMic() {
    clearTimeout(micIdle);
    micIdle = setTimeout(coolMic, 90000);   // let the microphone go after 1.5 minutes without use
    if (micStream || !navigator.mediaDevices) return;
    try { micStream = await navigator.mediaDevices.getUserMedia({audio: true}); await wait(600); }  // first time only
    catch (e) { micStream = null; }
  }
  function coolMic() { if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; } }
  function vlog(entry) {
    try { fetch('/api/voice-log', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(entry), keepalive: true}); } catch (e) {}
  }
  async function listen(opts = {}) {
    if (test()) {
      const next = test().queue.shift();
      return Promise.resolve(typeof next === 'function' ? next() : (next ?? null)).then(v => (v == null ? [] : [].concat(v)));
    }
    if (!Rec) return [];
    await warmMic();
    return new Promise(resolve => {
      const rec = new Rec();
      currentRec = rec;
      const lang = opts.lang || 'en-IN';
      rec.lang = lang; rec.interimResults = true; rec.continuous = true; rec.maxAlternatives = 5;
      const t0 = performance.now(), ev = [];
      const mark = name => ev.push(name + '@' + Math.round(performance.now() - t0));
      let finals = [], interim = '', ready = false, quiet = null, done = false, error = '';
      const stop = () => { try { rec.stop(); } catch (e) {} };
      const go = () => {
        if (ready) return; ready = true;
        setTimeout(() => { if (done) return; beep(); mark('beep'); status('Listening... please speak now'); }, 350);
      };
      const settle = ms => { clearTimeout(quiet); quiet = setTimeout(stop, ms); };
      rec.onstart = () => mark('start');
      rec.onaudiostart = () => { mark('audio'); go(); };
      rec.onsoundstart = () => mark('sound');
      rec.onspeechstart = () => mark('speech');
      rec.onspeechend = () => mark('speechend');
      rec.onresult = e => {
        for (let i = e.resultIndex; i < e.results.length; i++) {
          const r = e.results[i];
          if (r.isFinal) { finals = finals.concat([...r].map(x => x.transcript)); mark('final:' + r[0].transcript); }
          else { interim = r[0].transcript; mark('interim:' + interim); }
        }
        status('Heard: ' + (finals[0] || interim));
        settle(finals.length ? 900 : 1800);   // stop shortly after the person pauses
      };
      rec.onnomatch = () => mark('nomatch');
      rec.onerror = e => { error = e.error || ''; mark('error:' + error); };
      rec.onend = () => {
        if (done) return; done = true;
        mark('end');
        clearTimeout(quiet); currentRec = null;
        const alts = [...new Set((finals.length ? finals : (interim ? [interim] : [])).map(x => x.trim()).filter(Boolean))];
        lastHeard = alts[0] || '';
        vlog({page: location.pathname, question: opts.name || '', lang, heard: alts, events: ev, warm: !!micStream});
        status(alts.length ? 'Heard: ' + alts[0] : (error === 'not-allowed' ? 'The microphone is blocked. Please allow it in the browser.' : ''));
        if (error === 'not-allowed') alert('Please allow the microphone for this site (the icon in the address bar), then try again.');
        resolve(alts);
      };
      try { rec.start(); } catch (e) { done = true; currentRec = null; resolve([]); return; }
      setTimeout(go, 1200);   // some browsers never report audiostart
      setTimeout(stop, 9000); // longest wait for an answer
    });
  }
  /** What to suggest when a one-word answer was not caught: the same answer inside a short sentence works better. */
  function sentenceHint(el) {
    const k = el.dataset.voice;
    const sel = k === 'select' && field(el.dataset.name);
    const yn = sel && [...sel.options].filter(o => o.value !== '').map(o => o.text.toLowerCase()).sort().join() === 'no,yes';
    if (k === 'yesno' || yn) return 'For example, say: my answer is yes.';
    if (k === 'number') return 'For example, say: the answer is five.';
    if (k === 'help') return 'For example, say: she does it without help.';
    return 'Please say it again, a little slower.';
  }

  // ------------------------------------------------------------------ understanding answers
  const UNITS = {zero: 0, oh: 0, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10,
    eleven: 11, twelve: 12, thirteen: 13, fourteen: 14, fifteen: 15, sixteen: 16, seventeen: 17, eighteen: 18, nineteen: 19};
  const TENS = {twenty: 20, thirty: 30, forty: 40, fourty: 40, fifty: 50, sixty: 60, seventy: 70, eighty: 80, ninety: 90};
  const norm = t => String(t || '').toLowerCase().replace(/[^a-z0-9.' ]/g, ' ').replace(/\s+/g, ' ').trim();

  /** "seventy two" -> "72", "five feet four" -> "5 feet 4", "two thousand twenty six" -> "2026" */
  function wordsToDigits(text) {
    const out = []; let cur = null, total = 0;
    const flush = () => { if (cur !== null || total) out.push(String(total + (cur || 0))); cur = null; total = 0; };
    const tokens = String(text || '').toLowerCase().replace(/,/g, ' , ').replace(/[^a-z0-9.,' ]/g, ' ').split(/\s+/).filter(Boolean);
    for (const w of tokens) {
      const c = cur === null ? null : cur % 100;
      if (w in UNITS) {
        const v = UNITS[w];
        // a new number starts if this one already has a units digit or is a teen ("seventy two | sixty")
        if (c !== null && (c % 10 !== 0 || (c >= 10 && c < 20) || (v >= 10 && c !== 0))) flush();
        cur = (cur || 0) + v;
      } else if (w in TENS) {
        if (c !== null && c !== 0) flush();                 // "ninety three | eighty six"
        cur = (cur || 0) + TENS[w];
      } else if (w === 'hundred') { cur = (cur || 1) * 100; }
      else if (w === 'thousand') { total += (cur || 1) * 1000; cur = 0; }
      else if (w === 'and' && (cur !== null || total)) { /* "one hundred and five" */ }
      else { flush(); if (w !== ',') out.push(w); }
    }
    flush();
    return out.join(' ').replace(/(\d) point (\d)/g, '$1.$2');
  }
  const numbersIn = text => (wordsToDigits(text).match(/\d+(\.\d+)?/g) || []).map(Number);
  const has = (text, words) => words.some(w => new RegExp(`\\b${w}\\b`).test(text));
  const YES = ['yes', 'yeah', 'yep', 'yup', 'haan', 'han', 'ha', 'ji', 'correct', 'right', 'sure', 'true', 'of course', 'definitely', 'he does', 'she does', 'they do', 'i do'];
  const NO = ['no', 'nope', 'nah', 'nahi', 'nahin', 'not', 'never', 'none', "don't", 'dont', "doesn't", 'doesnt', 'false'];
  const SKIP = ['skip', "don't know", 'dont know', 'do not know', 'not known', 'no idea', 'not sure', 'pass'];

  // Short answers are often heard as a sound-alike word, especially with Indian-English recognition.
  // These are only used when the whole answer is one or two words.
  const YES_SOUND = ['s', 'es', 'yes', 'yash', 'yesh', 'yus', 'yees', 'yeas', 'jess', 'jas', 'ease', 'is', 'yas', 'yess', 'yea', 'ya', 'yah', 'yeh', 'jes', 'ok', 'okay', 'haa', 'haanji', 'hanji', 'theek'];
  const NO_SOUND = ['know', 'now', 'noh', 'nope', 'no no', 'naa', 'na', 'nai', 'noo', 'nu', 'nor', 'nahi', 'nako'];
  const NUM_SOUND = {to: 2, too: 2, tu: 2, for: 4, fore: 4, tree: 3, free: 3, won: 1, van: 1, ate: 8, tin: 10, nil: 0, nought: 0,
    naught: 0, nothing: 0, none: 0, zeero: 0, ziro: 0, jero: 0, zero: 0, o: 0, oh: 0, fiv: 5, sevan: 7, nein: 9, nyne: 9,
    shunya: 0, ek: 1, teen: 3, char: 4, chaar: 4, paanch: 5, panch: 5, chhe: 6, che: 6, saat: 7, aath: 8, nau: 9, das: 10};
  const short = t => t.split(' ').filter(Boolean).length <= 2;
  function shortNumber(t) {   // "zero", "free", "paanch", "no" on their own
    if (!short(t)) return undefined;
    for (const w of t.split(' ')) {
      if (w in NUM_SOUND) return NUM_SOUND[w];
      if (['no', 'nahi', 'never'].includes(w)) return 0;
    }
    return undefined;
  }
  /** A rough sound key: "shoe" and "shu" -> "sha", "clock" -> "klak", "lakh" -> "lak". */
  function soundKey(w) {
    let k = String(w).toLowerCase().replace(/[^a-z]/g, '');
    if (k.length > 3) k = k.replace(/(es|s)$/, '');
    if (k.length > 3) k = k.replace(/e$/, '');   // silent final e
    k = k.replace(/dge?/g, 'j').replace(/c(?=[eiy])/g, 's').replace(/ph/g, 'f').replace(/ck/g, 'k').replace(/kh/g, 'k').replace(/gh/g, 'g').replace(/[cq]/g, 'k')
         .replace(/x/g, 'ks').replace(/z/g, 's').replace(/w/g, 'v').replace(/y/g, 'i');
    k = k.replace(/[aeiou]+/g, 'a').replace(/(.)\1+/g, '$1');
    return k;
  }
  function lev(a, b) {
    const d = Array.from({length: a.length + 1}, (_, i) => [i]);
    for (let j = 1; j <= b.length; j++) d[0][j] = j;
    for (let i = 1; i <= a.length; i++) for (let j = 1; j <= b.length; j++)
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    return d[a.length][b.length];
  }
  function soundsLike(heard, word) {
    if (heard === word) return true;
    const a = soundKey(heard), b = soundKey(word);
    if (!a || !b) return false;
    if (a === b) return true;
    const n = Math.max(a.length, b.length);
    // one sound added or dropped ("lak" / "klak"), but not one sound changed ("brad" is not "braj")
    return (n >= 4 && a.length !== b.length && lev(a, b) <= 1) || (n >= 7 && lev(a, b) <= 2);
  }
  const FILLER = ['the', 'a', 'an', 'and', 'um', 'uh', 'er', 'words', 'word', 'are', 'is', 'were', 'was', 'i', 'think', 'remember', 'it', 'its', "it's", 'this', 'that', 'one', '1'];
  /** The expected word (from data-expect) that a heard word sounds like, or null. */
  function expected(el, heardWord) {
    const list = (el.dataset.expect || '').split(',').filter(Boolean);
    return list.find(w => heardWord === w) || list.find(w => soundsLike(heardWord, w)) || null;
  }

  function yesNo(t) {
    if (has(t, NO) && !has(t, ['not sure'])) return 0;   // "no" wins over "yes" in "yes, not really"
    if (has(t, YES)) return 1;
    if (short(t)) {
      const ws = t.split(' ');
      if (ws.some(w => NO_SOUND.includes(w))) return 0;
      if (ws.some(w => YES_SOUND.includes(w))) return 1;
    }
    return null;
  }
  function helpLevel(t, allowNa) {
    if (allowNa && (has(t, ['never did', 'never done', 'not applicable', 'never']) )) return 'na';
    if (has(t, ["can't", 'cant', 'cannot', 'unable', 'full help', 'fully', 'completely', 'totally', 'always needs', 'no way'])) return '0';
    if (has(t, ['some help', 'little help', 'bit of help', 'sometimes', 'partly', 'partial', 'with help', 'needs help', 'a little'])) return '1';
    if (has(t, ['without help', 'alone', 'by himself', 'by herself', 'by myself', 'themselves', 'independent', 'on their own', 'on his own', 'on her own', 'no help', 'fine', 'easily', 'yes'])) return '2';
    return null;
  }
  const SELECT_SYNONYMS = {
    female: ['female', 'woman', 'lady', 'girl', 'mother', 'wife'], male: ['male', 'man', 'boy', 'gent', 'gentleman', 'father', 'husband'],
    none: ['none', 'no school', 'no schooling', 'not studied', 'illiterate', 'never went to school', 'no education'],
    'high school': ['school', 'high school', 'tenth', '10th', 'twelfth', '12th', 'secondary'],
    "bachelor's": ['bachelor', 'graduate', 'graduation', 'degree', 'college'],
    higher: ['masters', 'master', 'post graduate', 'postgraduate', 'phd', 'doctorate', 'higher'],
    myself: ['myself', 'me', 'self', 'mine', 'for me'], mother: ['mother', 'mom', 'mum', 'mummy', 'amma', 'maa', 'ammi'],
    father: ['father', 'dad', 'daddy', 'papa', 'appa', 'abba', 'pitaji'], grandparent: ['grandparent', 'grandmother', 'grandfather', 'grandma', 'grandpa', 'nani', 'dadi', 'nana', 'dada', 'paati', 'thatha'],
    'husband / wife / partner': ['husband', 'wife', 'partner', 'spouse'],
  };
  function matchOption(select, t) {
    const opts = [...select.options].filter(o => o.value !== '');
    const labels = opts.map(o => o.text.toLowerCase());
    // yes/no style selects
    if (labels.length === 2 && labels.includes('yes') && labels.includes('no')) {
      const yn = yesNo(t); return yn === null ? null : opts.find(o => o.text.toLowerCase() === (yn ? 'yes' : 'no'));
    }
    for (const key of Object.keys(SELECT_SYNONYMS)) {  // synonyms first (gender, education); female before male
      const o = opts.find(x => x.text.toLowerCase() === key);
      if (o && has(t, SELECT_SYNONYMS[key])) return o;
    }
    let best = null, bestScore = 0;
    for (const o of opts) {  // otherwise: the option sharing most words with what was said
      const words = norm(o.text).split(' ').filter(w => w.length > 2 && !['the', 'and', 'other'].includes(w));
      const stem = w => (w.length > 5 ? w.slice(0, -1) : w);   // "india" matches "Indian"
      const score = words.filter(w => t.includes(stem(w))).length / Math.max(words.length, 1);
      if (score > bestScore) { best = o; bestScore = score; }
    }
    if (best) return best;
    if (has(t, ['other', 'prefer not'])) return opts.find(o => /other/i.test(o.text)) || null;
    return null;
  }

  // ------------------------------------------------------------------ applying answers
  const field = name => form.querySelector(`[name="${name}"]`);
  function fire(el) { el.dispatchEvent(new Event('input', {bubbles: true})); el.dispatchEvent(new Event('change', {bubbles: true})); }
  function setValue(name, v) { const el = field(name); if (el) { el.value = v; fire(el); } return el; }
  function clickRadio(name, value) {
    const r = form.querySelector(`input[name="${name}"][value="${value}"]`);
    if (r) { r.click(); fire(r); }
    return r;
  }
  const cap = s => s.charAt(0).toUpperCase() + s.slice(1);
  const wait = ms => new Promise(r => setTimeout(r, ms));

  /** Interpret transcript t for item el. Returns {ok, confirm} or {ok:false, hint}. */
  async function apply(el, raw) {
    const kind = el.dataset.voice, name = el.dataset.name, t = norm(raw);
    const optional = el.dataset.optional === '1';
    if (optional && has(t, SKIP)) { if (name) setValue(name, ''); return {ok: true, confirm: 'Okay, skipped.'}; }
    switch (kind) {
      case 'yesno': {
        if (has(t, SKIP)) return {ok: false, hint: 'If you are not sure, please say no.'};
        const v = yesNo(t); if (v === null) return {ok: false, hint: 'Please say yes or no.'};
        clickRadio(name, v); return {ok: true, confirm: v ? 'Yes.' : 'No.'};
      }
      case 'help': {
        const v = helpLevel(t, el.dataset.na === '1');
        if (v === null) return {ok: false, hint: 'Please say: without help, some help, or cannot do it.'};
        clickRadio(name, v);
        return {ok: true, confirm: {'2': 'Does it without help.', '1': 'Needs some help.', '0': 'Cannot do it.', na: 'Never did this task.'}[v]};
      }
      case 'select': {
        const o = matchOption(field(name), t);
        if (!o) return {ok: false, hint: 'Please say one of the choices.'};
        setValue(name, o.value); return {ok: true, confirm: o.text + '.'};
      }
      case 'number': {
        const n = numbersIn(t)[0] ?? shortNumber(t), input = field(name);
        if (n === undefined) return {ok: false, hint: 'Please say a number.'};
        const lo = parseFloat(input.min), hi = parseFloat(input.max);
        if (name === 'Age' && n < 18) { setValue(name, n); return {ok: true, confirm: '', stop: true}; }
        if (!isNaN(lo) && (n < lo || n > hi)) return {ok: false, hint: `Please say a number between ${lo} and ${hi}.`};
        setValue(name, n); return {ok: true, confirm: String(n) + '.'};
      }
      case 'text': {
        if (el.dataset.expect) {   // picture naming: "a clock" heard as "1 lakh", "pencil" as "pensil"
          const ws = t.split(' ').filter(w => w && !FILLER.includes(w));
          const hit = ws.map(w => expected(el, w)).find(Boolean) || expected(el, ws.join(''));
          if (hit) { setValue(name, cap(hit)); return {ok: true, confirm: cap(hit) + '.'}; }
        }
        const v = cap(String(raw).trim()); if (!v) return {ok: false, hint: 'Please say it again.'};
        setValue(name, v); return {ok: true, confirm: v + '.'};
      }
      case 'age': {  // profile: age in years (switches the form to "enter age")
        const n = numbersIn(t)[0];
        if (n === undefined) return {ok: false, hint: 'Please say the age in years.'};
        if (n < 18) return {ok: true, confirm: '', stop: true};
        if (n > 105) return {ok: false, hint: 'Please say the age in years.'};
        clickRadio('age_mode', 'age'); setValue('age_years', n);
        return {ok: true, confirm: n + ' years.'};
      }
      case 'height2': {  // profile: height in any unit, understood by the server, fields filled in that unit
        const d = await (await fetch('/api/height?text=' + encodeURIComponent(wordsToDigits(t)))).json();
        if (d.error) return {ok: false, hint: d.error.replace(/Try .*$/, 'Say it like five feet four inches, or one hundred and sixty five centimetres.')};
        setValue('height_unit', d.unit);
        if (d.unit === 'ft_in') { setValue('height_ft', d.ft); setValue('height_inch', d.inch); }
        else setValue('height_value', d.value);
        await wait(500);
        return {ok: true, confirm: 'Understood as ' + d.text.replace('=', 'which is') + '.'};
      }
      case 'height': case 'weight': {
        let v = wordsToDigits(t).replace(/\b(kilo ?grams?|kilos?|kgs?)\b/, 'kg').replace(/\b(pounds?|lbs?)\b/, 'lb');
        if (kind === 'weight' && /\blb\b/.test(v)) setValue('weight_unit', 'lb');
        if (kind === 'weight' && /\bkg\b/.test(v)) setValue('weight_unit', 'kg');
        setValue(name, v.replace(/\b(kg|lb)\b/, '').trim() || v);
        await wait(700);  // the page asks the server how it understood the value
        const note = document.getElementById(kind + 'Note');
        if (note && note.classList.contains('text-red-600')) return {ok: false, hint: note.textContent};
        return {ok: true, confirm: note ? note.textContent.replace('=', 'which is') + '.' : v};
      }
      case 'words3': {
        const words = t.split(' ').filter(w => w && !FILLER.includes(w)).map(w => expected(el, w) || w).slice(0, 3);
        if (!words.length) return {ok: false, hint: 'Please say the words you remember.'};
        name.split(',').forEach((n, i) => setValue(n, words[i] || ''));
        return {ok: true, confirm: 'I heard ' + words.join(', ') + '.'};
      }
      case 'numbers5': {
        const nums = numbersIn(t).slice(0, 5);
        if (!nums.length) return {ok: false, hint: 'Please say the numbers.'};
        name.split(',').forEach((n, i) => setValue(n, nums[i] ?? ''));
        return {ok: true, confirm: 'I heard ' + nums.join(', ') + '.'};
      }
      case 'buttons': {
        const btn = [...document.querySelectorAll(el.dataset.buttons)].find(b => t.includes((b.dataset.read || b.textContent).toLowerCase()));
        if (!btn) return {ok: false, hint: 'Please say one of the words.'};
        btn.click(); return {ok: true, confirm: btn.textContent.trim() + '.'};
      }
      case 'command': {
        if (yesNo(t) === 1 || has(t, ['start', 'ready', 'begin', 'okay', 'ok', 'go', 'next', 'continue'])) {
          gclick(document.querySelector(el.dataset.click)); return {ok: true, confirm: ''};
        }
        return {ok: false, hint: 'Say start when you are ready.'};
      }
      case 'gate': {
        const v = yesNo(t); if (v === null) return {ok: false, hint: 'Please say yes or no.'};
        return {ok: true, confirm: '', gate: v};
      }
    }
    return {ok: false, hint: ''};
  }

  /** Try each of the recogniser's guesses until one is a valid answer. */
  async function understand(el, alts) {
    alts = alts.slice(0, 4);
    if (el.dataset.expect) {   // prefer the guess containing the most expected words (brain check)
      const hits = a => norm(a).split(' ').filter(w => w && expected(el, w)).length;
      alts = alts.map((a, i) => [a, hits(a), i]).sort((x, y) => y[1] - x[1] || x[2] - y[2]).map(x => x[0]);
    }
    let last = {ok: false, hint: ''};
    for (const a of alts) {
      const r = await apply(el, a);
      if (r.ok) return r;
      last = r;
    }
    return last;
  }
  const saidYes = alts => alts.map(yesNo).find(v => v !== null) === 1;
  function showHeard(el, alts, ok) {   // "Heard: ..." under the question
    const h = el.querySelector('.vheard');
    if (!h) return;
    h.textContent = alts.length ? 'Heard: "' + alts[0] + '"' + (ok ? '' : ' (not understood)') : 'Nothing was heard. Please try again, a little closer to the microphone.';
    h.classList.toggle('vheard-bad', !ok);
  }

  // ------------------------------------------------------------------ asking one question
  const visible = el => !el.closest('.hidden') && !el.classList.contains('hidden');
  function answered(el) {
    const kind = el.dataset.voice, name = el.dataset.name;
    if (['command', 'gate', 'buttons'].includes(kind)) return false;
    if (kind === 'yesno' || kind === 'help') return !!form.querySelector(`input[name="${name}"]:checked`);
    if (kind === 'age' || kind === 'height2')   // any visible box inside already filled
      return [...el.querySelectorAll('input:not([type=radio])')].some(i => visible(i) && i.value !== '');
    return name.split(',').every(n => field(n) && field(n).value !== '');
  }
  async function ask(el, myRun) {
    document.getElementById('voice_used').value = '1';   // recorded with the assessment
    el.classList.add('voice-active');
    el.scrollIntoView({behavior: 'smooth', block: 'center'});
    let result = {ok: false};
    await say(el.dataset.say);
    for (let attempt = 0; attempt < 3 && myRun === runId; attempt++) {
      const heard = await listen({name: el.dataset.name, lang: attempt === 1 ? 'en-US' : 'en-IN'});
      if (myRun !== runId) break;
      if (!heard.length) { showHeard(el, heard, false); await say('Sorry, I did not catch that. ' + sentenceHint(el)); continue; }
      result = await understand(el, heard);
      showHeard(el, heard, result.ok);
      if (result.ok) { if (result.confirm) await say(result.confirm); break; }
      await say('Sorry, I heard ' + heard[0] + '. ' + result.hint);
    }
    if (!result.ok && myRun === runId) await say("Let's leave this one for now.");
    el.classList.remove('voice-active');
    return result;
  }

  // ------------------------------------------------------------------ guided mode
  // A form is either a step-by-step wizard (sections with class "step") or a single page.
  const steps = () => [...form.querySelectorAll('.step')];
  const currentStep = () => steps().find(s => !s.classList.contains('hidden')) || form;
  async function runStep() {
    const myRun = ++runId;
    const step = currentStep();
    if (!step || !guideOn) return;
    const wizard = step !== form;
    const brain = step.dataset.brain === '1';
    if (wizard && step.dataset.say) await say(step.dataset.say);
    const scope = brain ? step.querySelector('.bc:not(.hidden)') : step;
    if (brain && scope && scope.dataset.say) await say(scope.dataset.say);
    if (brain && scope && scope.dataset.afterSay) gclick(document.querySelector(scope.dataset.afterSay));
    const items = [...scope.querySelectorAll('[data-voice]')].filter(el => visible(el) && !answered(el));
    const announced = new Set();
    for (const el of items) {
      if (myRun !== runId || !guideOn) return;
      if (!visible(el)) continue;  // an earlier answer may have hidden it
      const section = el.parentElement.closest('[data-say]');   // single-page forms: introduce each section once
      if (!wizard && section && section !== form && !announced.has(section)) { announced.add(section); await say(section.dataset.say); }
      const r = await ask(el, myRun);
      if (r.stop) {  // a child: explain and stop
        await say(document.getElementById('childMsg').textContent);
        return stopGuide();
      }
      if (r.gate === 0) { gclick(step.querySelector('[data-skip]')); await wait(300); return runStep(); }  // no health numbers
    }
    if (myRun !== runId || !guideOn) return;

    if (brain) {  // brain check: move through its tasks
      if (scope && scope.classList.contains('hidden')) return runStep();  // an answer opened the next task
      const panel = scope;
      const key = panel && panel.dataset.bc;
      if (key === 'done') { /* fall through to the step end */ }
      else if (panel && panel.dataset.wait === '1') { await say(panel.dataset.waitSay || 'When you have finished, tap Continue.'); return; }
      else if (panel) {
        const btn = panel.querySelector('[data-bc-next], [data-bc-finish]');
        if (btn) { gclick(btn); await wait(200); return runStep(); }
        return;
      }
    }
    if (!wizard || step.dataset.final === '1') {   // last part: submit after a spoken yes
      await say(step.dataset.final === '1' ? 'Everything is filled in. Say yes to get the result.'
                                           : 'All the questions are answered. Say yes to save.');
      const t = (await listen()).map(norm);
      if (myRun !== runId) return;
      if (saidYes(t)) { gclick(form.querySelector('[data-final-submit]') || form.querySelector('[type=submit]')); }
      else await say('Okay. You can check the answers and press the button when ready.');
      return;
    }
    await say('This part is done. Say yes to go to the next part.');
    const t = (await listen()).map(norm);
    if (myRun !== runId) return;
    if (saidYes(t) || t.some(x => has(x, ['next', 'continue', 'go']))) {
      gclick(document.getElementById('nextBtn'));
      await wait(300);
      if (currentStep() === step) { await say('Some answers are still missing. Let us go through them.'); }
      return runStep();
    }
    await say('Okay. Tap Next when you are ready.');
  }

  function startGuide() {
    if (!canListen) { say('This browser cannot listen. Please use Google Chrome, Microsoft Edge or Safari.'); }
    guideOn = true;
    document.body.classList.add('voice-on');
    updateBar();
    say('Voice help is on. I will read each question. After the beep, please say your answer.').then(runStep);
  }
  function stopGuide() {
    guideOn = false; runId++;
    if (canSpeak && !test()) speechSynthesis.cancel();
    if (currentRec) try { currentRec.abort(); } catch (e) {}
    coolMic();
    document.querySelectorAll('.voice-active').forEach(e => e.classList.remove('voice-active'));
    status(''); updateBar();
  }

  // re-run the guide whenever the visible step or brain-check task changes
  document.addEventListener('click', e => {
    if (!guideOn) return;
    const b = e.target.closest('#nextBtn, #backBtn, [data-skip], [data-bc-next], [data-bc-start], [data-bc-finish]');
    if (b && !guidedClick) setTimeout(runStep, 350);   // the person tapped; the guide's own clicks call runStep directly
  });

  // ------------------------------------------------------------------ controls on the page
  const bar = document.getElementById('voiceBar');
  function status(text) { const s = document.getElementById('voiceStatus'); if (s) { s.textContent = text; s.classList.toggle('hidden', !text); } }
  function updateBar() {
    document.getElementById('voiceStart').classList.toggle('hidden', guideOn);
    document.getElementById('voiceStop').classList.toggle('hidden', !guideOn);
  }
  if (bar) {
    if (!canSpeak && !canListen) { bar.querySelector('[data-unsupported]').classList.remove('hidden'); }
    else if (!Rec) { bar.querySelector('[data-no-listen]').classList.remove('hidden'); }
    document.getElementById('voiceStart').addEventListener('click', startGuide);
    document.getElementById('voiceStop').addEventListener('click', stopGuide);
  }
  // ------------------------------------------------------------------ per-question buttons
  // "Read aloud" only speaks: the question, any help text and the answer choices. It never opens the microphone.
  // "Speak my answer" opens the microphone for this one question (after a short prompt and a beep).
  // Either button first stops the step-by-step guide, so the guide cannot start listening in the background.
  const INSTRUCTION = /\s*(please )?(say|answer) (yes or no|skip|zero|never)\b[^.?!]*[.?!]?|\s*if (they never did this task|you don't know|it is not in the report)[^.?!]*[.?!]?/gi;
  function choices(el) {
    const kind = el.dataset.voice;
    let labels = [];
    if (kind === 'yesno' || kind === 'help')
      labels = [...el.querySelectorAll('input[type=radio]')].map(r => (r.closest('label') || r).textContent.trim());
    else if (kind === 'select') {
      const sel = field(el.dataset.name);
      labels = sel ? [...sel.options].filter(o => o.value !== '' && !o.disabled).map(o => o.text.trim()) : [];
    }
    labels = labels.filter(Boolean);
    if (!labels.length) return '';
    return ' The choices are: ' + labels.slice(0, -1).join(', ') + (labels.length > 1 ? ', and ' : '') + labels[labels.length - 1] + '.';
  }
  function readText(el) {
    return ((el.dataset.say || '').replace(INSTRUCTION, '').trim() + choices(el)).trim();
  }
  const PROMPTS = {yesno: 'Please say yes or no after the beep.',
                   help: 'Please say: without help, some help, or cannot do it, after the beep.',
                   select: 'Please say your choice after the beep.',
                   number: 'Please say the number after the beep.'};
  function stopAll() {  // stop the guide, any speech and any open microphone
    if (guideOn) stopGuide(); else { runId++; if (canSpeak && !test()) speechSynthesis.cancel(); if (currentRec) try { currentRec.abort(); } catch (e) {} }
    document.querySelectorAll('.voice-active').forEach(e => e.classList.remove('voice-active'));
    document.querySelectorAll('.vbtn[data-busy]').forEach(b => { b.removeAttribute('data-busy'); b.querySelector('span').textContent = b.dataset.label; });
  }
  function busy(btn, text) { btn.setAttribute('data-busy', '1'); btn.querySelector('span').textContent = text; }
  async function readAloud(el, btn) {
    stopAll();
    const myRun = runId;
    busy(btn, 'Reading...');
    await say(readText(el));
    if (myRun === runId) stopAll();
  }
  async function answerOne(el, btn) {
    stopAll();
    const myRun = ++runId;
    document.getElementById('voice_used').value = '1';
    el.classList.add('voice-active');
    let result = {ok: false};
    const sel = el.dataset.voice === 'select' && field(el.dataset.name);
    const yesNoSelect = sel && [...sel.options].filter(o => o.value !== '').map(o => o.text.toLowerCase()).sort().join() === 'no,yes';
    const prompt = yesNoSelect ? PROMPTS.yesno : (PROMPTS[el.dataset.voice] || 'Please say your answer after the beep.');
    await say((el.dataset.optional === '1' ? 'Say skip if you do not know. ' : '') + prompt);
    for (let attempt = 0; attempt < 3 && myRun === runId; attempt++) {
      busy(btn, 'Listening... speak now');
      const heard = await listen({name: el.dataset.name, lang: attempt === 1 ? 'en-US' : 'en-IN'});
      if (myRun !== runId) break;
      busy(btn, 'Checking...');
      if (!heard.length) { showHeard(el, heard, false); await say('Sorry, I did not catch that. ' + sentenceHint(el)); continue; }
      result = await understand(el, heard);
      showHeard(el, heard, result.ok);
      if (result.ok) { if (result.confirm) await say(result.confirm); break; }
      await say('Sorry, I heard ' + heard[0] + '. ' + result.hint);
    }
    if (result.stop && myRun === runId) await say(document.getElementById('childMsg').textContent);
    if (myRun === runId) stopAll();
  }
  const ICON_SPEAKER = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 5L6 9H2v6h4l5 4V5zM15.5 8.5a5 5 0 010 7M19 5a10 10 0 010 14"/></svg>';
  const ICON_MIC = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="2" width="6" height="12" rx="3"/><path d="M5 11a7 7 0 0014 0M12 18v4"/></svg>';
  if (canSpeak || canListen) document.querySelectorAll('[data-voice]').forEach(el => {
    if (['command', 'gate'].includes(el.dataset.voice)) return;
    const tools = document.createElement('div');
    tools.className = 'voice-tools';
    tools.innerHTML = (canSpeak || test() ? `<button type="button" class="vbtn" data-act="read" data-label="Read aloud" title="Hear this question and its choices">${ICON_SPEAKER}<span>Read aloud</span></button>` : '')
      + (canListen ? `<button type="button" class="vbtn" data-act="answer" data-label="Speak my answer" title="Answer this question by speaking">${ICON_MIC}<span>Speak my answer</span></button>` : '');
    tools.insertAdjacentHTML('beforeend', '<span class="vheard" aria-live="polite"></span>');
    const rd = tools.querySelector('[data-act=read]'), ans = tools.querySelector('[data-act=answer]');
    if (rd) rd.addEventListener('click', e => { e.preventDefault(); rd.hasAttribute('data-busy') ? stopAll() : readAloud(el, rd); });
    if (ans) ans.addEventListener('click', e => { e.preventDefault(); ans.hasAttribute('data-busy') ? stopAll() : answerOne(el, ans); });
    el.appendChild(tools);
  });

  // expose for tests and for the result page
  window.voice = {say, startGuide, stopGuide, wordsToDigits, yesNo, helpLevel, readText, isOn: () => guideOn};
})();

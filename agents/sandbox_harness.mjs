/**
 * Runtime sandbox harness for generated browser code.
 *
 * There is no real browser available in this environment, so we execute the
 * page's inline JavaScript under a minimal DOM/Canvas/timer stub and capture
 * any error that a real page would throw *on load* — the class of defect the
 * static verifier cannot see (a function called during init but never defined,
 * a throw inside a DOMContentLoaded handler, a bad property access on a
 * stubbed element, etc.).
 *
 * This is NOT a correctness oracle — it cannot tell whether a game "plays
 * well". It answers a narrower, high-value question: *does the code blow up
 * the moment it runs?* That catches the most common one-shot-generation
 * failure that slips past static analysis.
 *
 * Usage:  node sandbox_harness.mjs <path-to-js-file>
 * Output: a single JSON line: {"errors": [...], "logs": [...], "timedOut": bool}
 *
 * The stubs deliberately return chainable proxies for unknown DOM access so a
 * page that merely *touches* many elements doesn't fail spuriously — we only
 * surface genuine JS errors (ReferenceError, TypeError from calling a
 * non-function, explicit throws), not "this selector found nothing".
 */

import { readFileSync } from "node:fs";

const errors = [];
const logs = [];

// ---------------------------------------------------------------------------
// Minimal DOM / Canvas / browser API stubs
// ---------------------------------------------------------------------------

// A chainable proxy standing in for any DOM element. Property reads return
// callable proxies (so `el.foo.bar()` never throws), and the common element
// methods are real no-ops. getContext returns a canvas-2d stub.
function makeEl() {
  const target = function () {};
  return new Proxy(target, {
    get(_t, p) {
      switch (p) {
        case "getContext":
          return () => ctx2d;
        case "style":
          return styleStub;
        case "classList":
          return classListStub;
        case "dataset":
          return {};
        case "children":
        case "childNodes":
          return [];
        case "length":
          return 0;
        case "value":
          return "";
        case "textContent":
        case "innerHTML":
        case "innerText":
        case "id":
        case "className":
          return "";
        case "parentNode":
        case "parentElement":
          return makeEl();
        case "addEventListener":
        case "removeEventListener":
        case "appendChild":
        case "removeChild":
        case "insertBefore":
        case "setAttribute":
        case "removeAttribute":
        case "getAttribute":
        case "focus":
        case "blur":
        case "click":
        case "play":
        case "pause":
        case "load":
        case "remove":
        case "scrollIntoView":
        case "getBoundingClientRect":
          return () => (p === "getBoundingClientRect"
            ? { top: 0, left: 0, right: 0, bottom: 0, width: 0, height: 0, x: 0, y: 0 }
            : undefined);
        case Symbol.toPrimitive:
          return () => "";
        case "then": // never be mistaken for a thenable
          return undefined;
        default:
          return makeEl();
      }
    },
    set() {
      return true;
    },
    apply() {
      return makeEl();
    },
    has() {
      return true;
    },
  });
}

const styleStub = new Proxy({}, { get: () => "", set: () => true });
const classListStub = {
  add() {}, remove() {}, toggle() {}, contains() { return false; }, replace() {},
};

// Canvas 2D context stub — every method is a no-op, every property read is safe.
const ctx2d = new Proxy(
  { canvas: { width: 800, height: 600 } },
  {
    get(t, p) {
      if (p in t) return t[p];
      if (p === "measureText") return () => ({ width: 0 });
      if (p === "createLinearGradient" || p === "createRadialGradient")
        return () => ({ addColorStop() {} });
      if (p === "getImageData")
        return () => ({ data: new Uint8ClampedArray(4) });
      return () => {};
    },
    set() { return true; },
  }
);

// DOMContentLoaded / load handlers are queued and fired after the main script
// body runs, mimicking real page lifecycle.
const loadQueue = [];

const documentStub = {
  getElementById: () => makeEl(),
  getElementsByClassName: () => [],
  getElementsByTagName: () => [],
  querySelector: () => makeEl(),
  querySelectorAll: () => [],
  createElement: () => makeEl(),
  createElementNS: () => makeEl(),
  createTextNode: () => makeEl(),
  addEventListener: (ev, fn) => {
    if (ev === "DOMContentLoaded" && typeof fn === "function") loadQueue.push(fn);
  },
  removeEventListener: () => {},
  body: makeEl(),
  head: makeEl(),
  documentElement: makeEl(),
  cookie: "",
  title: "",
  readyState: "complete",
};

// ---------------------------------------------------------------------------
// Global environment
// ---------------------------------------------------------------------------

// Some of these (e.g. navigator in Node 21+) are read-only globals; assign
// defensively so the harness never crashes just wiring up the environment.
function setGlobal(name, value) {
  try {
    globalThis[name] = value;
  } catch {
    try {
      Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
    } catch {
      /* give up on this one global — not worth failing the whole run */
    }
  }
}

setGlobal("document", documentStub);
setGlobal("window", globalThis);
setGlobal("self", globalThis);
setGlobal("navigator", { userAgent: "sandbox", language: "en-US", platform: "sandbox" });
setGlobal("location", { href: "about:blank", hash: "", search: "", pathname: "/", reload() {} });
setGlobal("history", { pushState() {}, replaceState() {}, back() {}, forward() {} });

setGlobal("requestAnimationFrame", () => 0); // do NOT actually loop
setGlobal("cancelAnimationFrame", () => {});
// Timers are stubbed to no-ops so the harness exits promptly and game loops
// don't run forever; we only care about the synchronous load path.
setGlobal("setTimeout", () => 0);
setGlobal("setInterval", () => 0);
setGlobal("clearTimeout", () => {});
setGlobal("clearInterval", () => {});

setGlobal("alert", () => {});
setGlobal("confirm", () => true);
setGlobal("prompt", () => null);
setGlobal("localStorage", {
  _d: {},
  getItem(k) { return this._d[k] ?? null; },
  setItem(k, v) { this._d[k] = String(v); },
  removeItem(k) { delete this._d[k]; },
  clear() { this._d = {}; },
});
setGlobal("sessionStorage", {
  _d: {},
  getItem(k) { return this._d[k] ?? null; },
  setItem(k, v) { this._d[k] = String(v); },
  removeItem(k) { delete this._d[k]; },
  clear() { this._d = {}; },
});

const realConsole = console;
setGlobal("console", {
  log: (...a) => logs.push(a.map(String).join(" ")),
  info: (...a) => logs.push(a.map(String).join(" ")),
  warn: (...a) => logs.push(a.map(String).join(" ")),
  error: (...a) => {
    const msg = a.map(String).join(" ");
    logs.push(msg);
    errors.push("console.error: " + msg);
  },
  debug: () => {},
});

setGlobal("Image", function () { return makeEl(); });
setGlobal("Audio", function () { return makeEl(); });
setGlobal("fetch", () => Promise.resolve({ ok: true, json: () => Promise.resolve({}), text: () => Promise.resolve("") }));
setGlobal("AudioContext", function () {
  return new Proxy({}, { get: () => () => new Proxy({}, { get: () => () => {} }) });
});
setGlobal("webkitAudioContext", globalThis.AudioContext);

// ---------------------------------------------------------------------------
// Execute the user's JS
// ---------------------------------------------------------------------------

let userCode = "";
try {
  userCode = readFileSync(process.argv[2], "utf-8");
} catch (e) {
  realConsole.log(JSON.stringify({ errors: ["harness: could not read code file: " + e], logs: [], timedOut: false }));
  process.exit(0);
}

let timedOut = false;
const guard = setTimeout(() => {
  timedOut = true;
  realConsole.log(JSON.stringify({ errors, logs, timedOut }));
  process.exit(0);
}, 4000);
// Do not let the guard keep the event loop alive on its own.
if (guard.unref) guard.unref();

try {
  // Indirect eval in module scope; `const`/`let`/`function` at top level of the
  // user code become locals of this eval, which is fine — we only need to run it.
  const runner = new Function(userCode);
  runner();
  // Fire queued DOMContentLoaded / load handlers (page "finished loading").
  for (const fn of loadQueue) {
    try {
      fn();
    } catch (e) {
      errors.push(errName(e));
    }
  }
} catch (e) {
  errors.push(errName(e));
}

clearTimeout(guard);
realConsole.log(JSON.stringify({ errors, logs, timedOut }));

function errName(e) {
  if (e instanceof Error) {
    const first = (e.stack || "").split("\n").find((l) => /at /.test(l));
    return `${e.name}: ${e.message}` + (first ? ` (${first.trim()})` : "");
  }
  return String(e);
}

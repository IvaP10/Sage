import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

const source = (await readFile(new URL("background.js", import.meta.url), "utf8")).replaceAll("export function", "function");

async function browser() {
  const listeners = {};
  const event = name => ({ addListener: callback => { listeners[name] = callback; } });
  const tab = { id: 7, windowId: 4, url: "https://example.com/start", status: "complete" };
  let document = "initial-document", mutations = 0;
  const replies = [];
  let reply;
  const port = { onMessage: event("message"), onDisconnect: event("disconnect"), disconnect() {},
    postMessage(value) { replies.push(value); reply?.(value); } };
  const chrome = {
    tabs: { onUpdated: event("updated"), onRemoved: event("removed"),
      async get() { return { ...tab }; },
      async update(id, change) { assert.equal(id, tab.id); mutations++; Object.assign(tab, change); document = `document-${mutations}`; listeners.updated(tab.id, { status: "loading" }); } },
    action: { onClicked: event("clicked"), async setBadgeText() {}, async setTitle() {} },
    runtime: { connectNative() { return port; } },
    scripting: { async executeScript() { return [{ frameId: 0, documentId: document, result: { url: tab.url, origin: new URL(tab.url).origin } }]; } },
  };
  vm.runInNewContext(source, { chrome, URL, Date, setTimeout, clearTimeout });
  await listeners.clicked(tab);
  return {
    get mutations() { return mutations; }, replies,
    async request(operation, payload = {}) {
      const response = new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(Error("Browser reply timed out")), 1000);
        reply = value => { clearTimeout(timer); resolve(value); };
      });
      listeners.message({ request_id: String(replies.length + 1), operation, payload, expires_at_unix_ms: Date.now() + 5000 });
      return response;
    },
  };
}
function grant(id = "one-use") {
  return { id, domain: "browser", policy_version: 2, remaining_uses: 0, revoked: false,
    expires_at: new Date(Date.now() + 5000).toISOString(),
    resource: { kind: "browser_origin", origin: "https://example.com" }, operations: ["network", "control"] };
}
const action = { type: "navigate_url", url: "https://example.com/next", new_tab: false };

test("each stale browser identity field prevents navigation", async () => {
  for (const field of ["tab_id", "window_id", "frame_id", "document_id", "navigation_generation", "origin", "url"]) {
    const b = await browser();
    const binding = (await b.request("binding")).data;
    binding[field] = typeof binding[field] === "number" ? binding[field] + 1 : "changed";
    const response = await b.request("execute", { action, capability: grant(), browser_target: binding });
    assert.equal(response.success, false, field);
    assert.equal(b.mutations, 0, field);
  }
});

test("a valid navigation returns the new document and consumes its grant", async () => {
  const b = await browser();
  const before = (await b.request("binding")).data;
  const first = await b.request("execute", { action, capability: grant(), browser_target: before });
  assert.equal(first.success, true);
  assert.equal(first.data.browser_target.url, action.url);
  assert.notEqual(first.data.browser_target.document_id, before.document_id);
  const replay = await b.request("execute", { action, capability: grant(), browser_target: first.data.browser_target });
  assert.equal(replay.success, false);
  assert.equal(b.mutations, 1);
});

test("unsupported operations, other origins, expired and revoked grants do not dispatch", async () => {
  for (const mutation of [
    payload => { payload.action = { type: "type_text", text: "private text" }; },
    payload => { payload.action = { ...action, url: "https://different.example/" }; },
    payload => { payload.capability.revoked = true; },
    payload => { payload.capability.expires_at = "invalid"; },
    payload => { payload.capability.expires_at = new Date(0).toISOString(); },
    payload => { payload.capability.policy_version = 1; },
  ]) {
    const b = await browser();
    const payload = { action, capability: grant(), browser_target: (await b.request("binding")).data };
    mutation(payload);
    assert.equal((await b.request("execute", payload)).success, false);
    assert.equal(b.mutations, 0);
  }
});

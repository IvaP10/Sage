import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";
import { randomUUID } from "node:crypto";

const source = (await readFile(new URL("background.js", import.meta.url), "utf8")).replaceAll("export function", "function");

async function browser() {
  const listeners = {};
  const event = name => ({ addListener: callback => { listeners[name] = callback; } });
  const tab = { id: 7, windowId: 4, url: "https://example.com/start", status: "complete" };
  let document = "initial-document", mutations = 0, referenceReads = 0, discoveryReads = 0, activeTabId = 7, nextIdentity, nextReference, nextDiscovery, nextUpdate;
  const replies = [];
  const waiters = new Map();
  const port = { onMessage: event("message"), onDisconnect: event("disconnect"), disconnect() {},
    postMessage(value) {
      replies.push(value);
      const key = `${value.request_id}:${value.operation || "result"}`;
      waiters.get(key)?.(value); waiters.delete(key);
    } };
  const chrome = {
    tabs: { onUpdated: event("updated"), onRemoved: event("removed"),
      async get() { return { ...tab }; },
      async query() { return activeTabId === 0 ? [] : [{ id: activeTabId }]; },
      async update(id, change) {
        assert.equal(id, tab.id); mutations++; Object.assign(tab, change); document = `document-${mutations}`;
        listeners.updated(tab.id, { status: "loading" });
        const barrier = nextUpdate; nextUpdate = undefined; if (barrier) await barrier();
      } },
    action: { onClicked: event("clicked"), async setBadgeText() {}, async setTitle() {} },
    runtime: { connectNative() { return port; } },
    scripting: { async executeScript(request) {
      const barrier = nextIdentity; nextIdentity = undefined; if (barrier) await barrier();
      if (request.args?.[1] === "reference") {
        const referenceBarrier = nextReference; nextReference = undefined; if (referenceBarrier) await referenceBarrier();
        referenceReads++;
        return [{ frameId: 0, documentId: document, result: { title: "Example", url: tab.url, selected_text: "selected", page_text: request.args[2].include_page_text ? "visible page" : "" } }];
      }
      if (request.args?.[1] === "discover") {
        const discoveryBarrier = nextDiscovery; nextDiscovery = undefined; if (discoveryBarrier) await discoveryBarrier();
        discoveryReads++;
        return [{ frameId: 0, documentId: document, result: { controls: [
          { role: "button", kind: "button", label: "Save", enabled: true, ancestors: ["Editor"] },
        ], truncated: false } }];
      }
      return [{ frameId: 0, documentId: document, result: { url: tab.url, origin: new URL(tab.url).origin } }];
    } },
  };
  vm.runInNewContext(source, { chrome, URL, Date, setTimeout, clearTimeout });
  await listeners.clicked(tab);
  function waitFor(id, operation = "result") {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { waiters.delete(`${id}:${operation}`); reject(Error("Browser reply timed out")); }, 2000);
      waiters.set(`${id}:${operation}`, value => { clearTimeout(timer); resolve(value); });
    });
  }
  function start(operation, payload = {}, id = randomUUID()) {
    const response = waitFor(id);
    listeners.message({ request_id: id, operation, payload, expires_at_unix_ms: Date.now() + 5000 });
    return { id, response };
  }
  function barrier(install) {
    let release, entered;
    const blocked = new Promise(resolve => { release = resolve; });
    const ready = new Promise(resolve => { entered = resolve; });
    install(() => { entered(); return blocked; });
    return { ready, release };
  }
  return {
    get mutations() { return mutations; }, get referenceReads() { return referenceReads; }, get discoveryReads() { return discoveryReads; }, replies, start,
    setActiveTab(id) { activeTabId = id; },
    holdIdentity() { return barrier(value => { nextIdentity = value; }); },
    holdReference() { return barrier(value => { nextReference = value; }); },
    holdDiscovery() { return barrier(value => { nextDiscovery = value; }); },
    holdUpdate() { return barrier(value => { nextUpdate = value; }); },
    async request(operation, payload = {}) { return start(operation, payload).response; },
    async cancel(id) {
      const response = waitFor(id, "cancel_ack");
      listeners.message({ request_id: id, operation: "cancel", expires_at_unix_ms: Date.now() + 5000 });
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

function runPageReference(elements, selectedText = "selected") {
  const privateElements = elements.filter(element => element.private);
  const range = { intersectsNode(element) { return element.private; } };
  const selection = {
    rangeCount: 1,
    anchorNode: { parentElement: elements[0] },
    focusNode: { parentElement: elements[0] },
    getRangeAt() { return range; },
    toString() { return selectedText; },
  };
  let cursor = 0;
  const document = {
    title: "Example",
    body: {},
    createTreeWalker() { return { nextNode() { return elements[cursor++] || null; } }; },
    querySelectorAll(selector) { return selector === "[data-sage-private]" ? privateElements : []; },
  };
  const event = { addListener() {} };
  const context = {
    chrome: { tabs: { onUpdated: event, onRemoved: event }, action: { onClicked: event }, runtime: {} },
    document,
    location: { origin: "https://example.com", href: "https://example.com/article" },
    NodeFilter: { SHOW_TEXT: 4 },
    getSelection: () => selection,
    getComputedStyle: () => ({ display: "block", visibility: "visible" }),
  };
  vm.runInNewContext(`${source}; globalThis.__pageOperation = pageOperation;`, context);
  return context.__pageOperation("https://example.com", "reference", { include_page_text: true });
}

function runPageDiscovery(elements) {
  const element = (tagName, attributes = {}, options = {}) => {
    const value = {
      tagName: tagName.toUpperCase(),
      disabled: options.disabled ?? false,
      readOnly: options.readOnly ?? false,
      isContentEditable: options.contentEditable ?? false,
      textNodes: options.textNodes ?? [],
      labels: options.labels,
      parentElement: null,
      getAttribute(name) { return Object.hasOwn(attributes, name) ? attributes[name] : null; },
      matches(selector) {
        const type = (attributes.type || "").toLowerCase();
        return (selector.includes('input[type="password"]') && this.tagName === "INPUT" && type === "password")
          || (selector.includes('input[autocomplete*="cc-"]') && String(attributes.autocomplete || "").includes("cc-"))
          || (selector.includes('input[autocomplete*="password"]') && String(attributes.autocomplete || "").includes("password"));
      },
      closest(selector) {
        let current = this;
        while (current) {
          if (selector.includes("data-sage-private") && current.attrs?.["data-sage-private"] !== undefined) return current;
          if (selector.includes("[hidden]") && current.attrs?.hidden !== undefined) return current;
          if (selector.includes('aria-hidden="true"') && current.attrs?.["aria-hidden"] === "true") return current;
          current = current.parentElement;
        }
        return null;
      },
      getClientRects() { return options.visible === false ? [] : [1]; },
      attrs: attributes,
    };
    return value;
  };
  const body = { tagName: "BODY", parentElement: null, attrs: {}, getAttribute() { return null; }, matches() { return false; }, closest() { return null; }, getClientRects() { return [1]; } };
  for (const item of elements) item.parentElement ||= body;
  const document = {
    body,
    title: "Ignored title",
    createTreeWalker(root) {
      const sequence = root === body ? elements : root.textNodes.map(text => ({ nodeValue: text, parentElement: root }));
      let cursor = 0;
      return { nextNode() { return sequence[cursor++] || null; } };
    },
  };
  const event = { addListener() {} };
  const context = {
    chrome: { tabs: { onUpdated: event, onRemoved: event }, action: { onClicked: event }, runtime: {} },
    document,
    location: { origin: "https://example.com", href: "https://example.com/app" },
    NodeFilter: { SHOW_TEXT: 4, SHOW_ELEMENT: 1 },
    getComputedStyle: () => ({ display: "block", visibility: "visible" }),
  };
  vm.runInNewContext(`${source}; globalThis.__pageOperation = pageOperation;`, context);
  return { result: context.__pageOperation("https://example.com", "discover", {}), element };
}

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

test("foreground references read only the active paired page and bound page text", async () => {
  const b = await browser();
  const result = await b.request("reference", { include_page_text: true });
  assert.equal(result.success, true);
  assert.equal(result.data.available, true);
  assert.equal(result.data.selected_text, "selected");
  assert.equal(result.data.page_text, "visible page");
  assert.equal(b.referenceReads, 1);

  b.setActiveTab(8);
  const stale = await b.request("reference", { include_page_text: true });
  assert.equal(stale.success, true);
  assert.equal(stale.data.available, false);
  assert.equal(b.referenceReads, 1);
});

test("browser interface discovery requires a stable foreground paired document", async () => {
  const b = await browser();
  b.setActiveTab(8);
  const inactive = await b.request("discover_interface");
  assert.equal(inactive.success, true);
  assert.equal(inactive.data.available, false);
  assert.equal(inactive.data.reason, "paired_tab_not_active");
  assert.equal(b.discoveryReads, 0);

  b.setActiveTab(7);
  const gate = b.holdDiscovery();
  const pending = b.start("discover_interface");
  await gate.ready;
  b.setActiveTab(8);
  gate.release();
  const changed = await pending.response;
  assert.equal(changed.success, true);
  assert.equal(changed.data.available, false);
  assert.equal(changed.data.reason, "active_tab_changed_during_observation");
  assert.equal(b.mutations, 0);

  b.setActiveTab(7);
  const current = await b.request("discover_interface");
  assert.equal(current.success, true);
  assert.equal(current.data.available, true);
  assert.equal(current.data.controls[0].label, "Save");
  assert.equal(b.mutations, 0);
});

test("passive DOM discovery excludes secure, hidden, private and editable contents", () => {
  const { element } = runPageDiscovery([]);
  const button = element("button", {}, { textNodes: ["Save"] });
  const secure = element("input", { type: "password", "aria-label": "Password" });
  const privateContainer = element("section", { "data-sage-private": "" });
  const privateButton = element("button", {}, { textNodes: ["Private action"] });
  privateButton.parentElement = privateContainer;
  const hiddenButton = element("button", {}, { textNodes: ["Hidden action"], visible: false });
  const emailButton = element("button", {}, { textNodes: ["person@example.com"] });
  const secretButton = element("button", { "aria-label": "api_key=sk-123456789012345" });
  const editable = element("div", { "aria-label": "Message", contenteditable: "true" }, { contentEditable: true, textNodes: ["draft that must not persist"] });
  const fixture = runPageDiscovery([button, secure, privateButton, hiddenButton, emailButton, secretButton, editable]).result;
  assert.equal(fixture.error, undefined);
  assert.deepEqual(Array.from(fixture.controls, control => control.label), ["Save", "Message"]);
  assert.equal(JSON.stringify(fixture).includes("draft that must not persist"), false);
  assert.equal(JSON.stringify(fixture).includes("person@example.com"), false);
  assert.equal(JSON.stringify(fixture).includes("sk-123456789012345"), false);
  assert.equal(fixture.truncated, false);
});

test("passive DOM discovery caps controls and marks a truncated scan", () => {
  const { element } = runPageDiscovery([]);
  const controls = Array.from({ length: 40 }, (_, index) => element("button", { "aria-label": `Control ${index}` }));
  const result = runPageDiscovery(controls).result;
  assert.equal(result.controls.length, 28);
  assert.equal(result.truncated, true);
});

test("reference collection discards private selected ranges and marked private page text", () => {
  const element = ({ text, private: isPrivate = false, visible = true } = {}) => {
    const value = {
      nodeValue: text,
      private: isPrivate,
      matches: () => false,
      closest: selector => isPrivate && selector.includes("data-sage-private") ? {} : null,
      getClientRects: () => visible ? [1] : [],
      parentElement: null,
    };
    value.parentElement = value;
    return value;
  };
  const publicNode = element({ text: "Public page text" });
  const privateNode = element({ text: "private account details", private: true });
  const hiddenNode = element({ text: "hidden text", visible: false });
  const result = runPageReference([publicNode, privateNode, hiddenNode]);
  assert.equal(result.selected_text, "");
  assert.equal(result.page_text, "Public page text");
});

test("foreground reference text has strict selection and page size limits", () => {
  const large = "x".repeat(10000);
  const node = {
    nodeValue: large,
    private: false,
    matches: () => false,
    closest: () => null,
    getClientRects: () => [1],
  };
  node.parentElement = node;
  const result = runPageReference([node], "s".repeat(5000));
  assert.equal(result.selected_text.length, 4000);
  assert.equal(result.page_text.length, 8000);
});

test("a tab switch during reference capture discards the observation", async () => {
  const b = await browser();
  const gate = b.holdReference();
  const pending = b.start("reference", { include_page_text: true });
  await gate.ready;
  b.setActiveTab(8);
  gate.release();
  const result = await pending.response;
  assert.equal(result.success, true);
  assert.equal(result.data.available, false);
});

test("Stop before admission prevents a browser effect", async () => {
  const b = await browser();
  const target = (await b.request("binding")).data;
  const id = randomUUID();
  await b.cancel(id);
  const result = await b.start("execute", { action, capability: grant(), browser_target: target }, id).response;
  assert.equal(result.success, false);
  assert.match(result.error, /cancelled before dispatch/);
  assert.equal(b.mutations, 0);
});

test("Stop bypasses a blocked identity read and cancels both active and queued requests", async () => {
  const b = await browser();
  const target = (await b.request("binding")).data;
  const gate = b.holdIdentity();
  const active = b.start("execute", { action, capability: grant("active"), browser_target: target });
  await gate.ready;
  const queued = b.start("execute", { action, capability: grant("queued"), browser_target: target });
  await b.cancel(active.id);
  await b.cancel(queued.id);
  gate.release();
  assert.equal((await active.response).success, false);
  assert.equal((await queued.response).success, false);
  assert.equal(b.mutations, 0);
});

test("Stop after navigation dispatch reports uncertainty rather than prevented navigation", async () => {
  const b = await browser();
  const target = (await b.request("binding")).data;
  const gate = b.holdUpdate();
  const active = b.start("execute", { action, capability: grant(), browser_target: target });
  await gate.ready;
  await b.cancel(active.id);
  assert.equal(b.mutations, 1);
  gate.release();
  const result = await active.response;
  assert.equal(result.success, false);
  assert.match(result.error, /after navigation was dispatched; the tab may have changed/);
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

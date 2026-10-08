// A user gesture pairs one tab. Every execution binds its prepared document.
let port, paired;
const generations = new Map();

chrome.tabs.onUpdated.addListener((id, change) => {
  if (change.status === "loading" || change.url) generations.set(id, (generations.get(id) || 0) + 1);
});
chrome.tabs.onRemoved.addListener(id => {
  generations.delete(id);
  if (paired?.tabId === id) { port?.disconnect(); port = undefined; paired = undefined; }
});
chrome.action.onClicked.addListener(async tab => {
  port?.disconnect(); port = undefined; paired = undefined;
  if (!Number.isInteger(tab.id) || !/^https?:/.test(tab.url || "")) return;
  generations.set(tab.id, (generations.get(tab.id) || 0) + 1);
  paired = { tabId: tab.id, windowId: tab.windowId, origin: new URL(tab.url).origin };
  const connection = chrome.runtime.connectNative("com.ivanpadeliya.sage.browser");
  port = connection;
  // Each connection owns its queue and revocation records. Stop is handled
  // outside that queue, including while an earlier Chrome API is awaiting.
  const state = { queue: Promise.resolve(), pending: 0, records: new Map(), usedGrants: new Map() };
  connection.onMessage.addListener(request => {
    if (port !== connection) return;
    try {
      const id = request.request_id, expiry = request.expires_at_unix_ms;
      if (typeof id !== "string" || !/^[0-9a-f-]{36}$/i.test(id) || !Number.isSafeInteger(expiry)) throw Error("Invalid browser request identity");
      for (const [key, record] of state.records) {
        if (!record.active && record.expiry < Date.now() - 60000) state.records.delete(key);
      }
      let record = state.records.get(id);
      if (request.operation === "cancel") {
        if (!record) {
          if (state.records.size >= 4096) throw Error("Browser cancellation capacity exceeded");
          record = { expiry, active: false, cancelled: true, admitted: false };
          state.records.set(id, record);
        }
        record.cancelled = true;
        connection.postMessage({ request_id: id, operation: "cancel_ack" });
        return;
      }
      if (record?.admitted) throw Error("Duplicate browser request");
      if (state.pending >= 32 || (!record && state.records.size >= 4096)) {
        connection.postMessage({ request_id: id, success: false, error: "Browser worker capacity exceeded", data: {} });
        return;
      }
      record ||= { expiry, cancelled: false };
      record.active = true; record.admitted = true;
      state.records.set(id, record); state.pending++;
      state.queue = state.queue.then(() => handle(request, connection, record, state.usedGrants))
        .finally(() => { record.active = false; state.pending--; }).catch(() => {});
    } catch (_) {
      // A revocation that cannot be retained closes its whole authority scope.
      if (port === connection) { port = undefined; paired = undefined; }
      connection.disconnect();
    }
  });
  connection.onDisconnect.addListener(() => {
    const message = chrome.runtime.lastError?.message;
    if (port !== connection) return;
    port = undefined; paired = undefined;
    chrome.action.setBadgeText({ text: "" });
    chrome.action.setTitle({ title: message || "Pair this tab with Sage" });
  });
  await chrome.action.setBadgeText({ text: "ON" });
});

export function sameTarget(a, b) {
  const fields = ["tab_id", "window_id", "frame_id", "document_id", "navigation_generation", "origin", "url"];
  return !!a && !!b && fields.every(key => a[key] !== undefined && a[key] === b[key]);
}

async function binding() {
  const session = paired;
  if (!session) throw Error("Pair a browser tab first");
  const generation = generations.get(session.tabId);
  const tab = await chrome.tabs.get(session.tabId);
  if (tab.windowId !== session.windowId || new URL(tab.url).origin !== session.origin) throw Error("Paired tab changed; pair it again");
  const results = await chrome.scripting.executeScript({ target: { tabId: tab.id, frameIds: [0] }, world: "ISOLATED", func: () => ({ url: location.href, origin: location.origin }) });
  const result = results[0];
  if (paired !== session || generation !== generations.get(session.tabId) || results.length !== 1 || result?.frameId !== 0 || !result.documentId
    || result.result?.url !== tab.url || result.result?.origin !== session.origin) throw Error("Document changed while observing its identity");
  return { tab_id: tab.id, window_id: tab.windowId, frame_id: 0, document_id: result.documentId,
    navigation_generation: generation, origin: session.origin, url: tab.url };
}

async function handle(request, replyPort, record, usedGrants) {
  if (replyPort !== port) return;
  const response = { request_id: request.request_id, success: false, data: {} };
  let dispatched = false;
  const checkAlive = () => {
    if (record.cancelled || replyPort !== port) throw Error(dispatched
      ? "Stop was requested after navigation was dispatched; the tab may have changed"
      : "Browser request cancelled before dispatch");
    if (!paired || request.expires_at_unix_ms <= Date.now()) throw Error(dispatched
      ? "Browser request expired after navigation was dispatched; the tab may have changed"
      : "Browser request is stale or unpaired");
  };
  try {
    checkAlive();
    const target = await binding();
    checkAlive();
    const payload = request.payload || {};
    if (request.operation === "binding") response.data = target;
    else if (request.operation === "discover_interface") {
      const active = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
      checkAlive();
      if (active.length !== 1 || active[0].id !== target.tab_id) {
        response.data = { available: false, reason: "paired_tab_not_active" };
      } else {
        const captured = await dom(target, "discover", {});
        checkAlive();
        const targetAfter = await binding();
        const activeAfter = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
        if (!sameTarget(target, targetAfter)) {
          response.data = { available: false, reason: "document_changed_during_observation" };
        } else if (activeAfter.length !== 1 || activeAfter[0].id !== target.tab_id) {
          response.data = { available: false, reason: "active_tab_changed_during_observation" };
        } else {
          response.data = { ...captured, available: true, browser_target: target };
        }
      }
    }
    else if (request.operation === "reference") {
      const active = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
      checkAlive();
      if (active.length !== 1 || active[0].id !== target.tab_id) {
        response.data = { available: false, reason: "paired_tab_not_active" };
      } else {
        const captured = await dom(target, "reference", payload);
        checkAlive();
        const activeAfter = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
        if (activeAfter.length !== 1 || activeAfter[0].id !== target.tab_id) {
          response.data = { available: false, reason: "active_tab_changed_during_observation" };
        } else {
          response.data = { ...captured, available: true, browser_target: target };
        }
      }
    }
    else if (request.operation === "observe") {
      const condition = payload.condition || {};
      if (condition.kind === "url_equals") response.data = { url: target.url, browser_target: target };
      else if (condition.kind === "element_present") response.data = await dom(target, "observe", condition);
      else throw Error("Unsupported browser observation");
    } else if (request.operation === "execute") {
      const action = payload.action || {}, grant = payload.capability || {};
      if (!sameTarget(target, payload.browser_target)) throw Error("Approved browser target is stale; prepare the action again");
      for (const [id, expiry] of usedGrants) if (expiry <= Date.now()) usedGrants.delete(id);
      if (grant.domain !== "browser" || grant.policy_version !== 2 || grant.remaining_uses !== 0 || grant.revoked || !grant.id
        || !Number.isFinite(Date.parse(grant.expires_at)) || Date.parse(grant.expires_at) <= Date.now()
        || grant.resource?.kind !== "browser_origin" || grant.resource.origin !== target.origin || usedGrants.has(grant.id)
        || !grant.operations?.includes("network") || !grant.operations?.includes("control")) throw Error("Browser capability does not match this session");
      if (action.type !== "navigate_url") throw Error("This browser operation has not passed feature qualification");
      const destination = new URL(action.url);
      if (destination.origin !== target.origin || !["http:", "https:"].includes(destination.protocol) || destination.username || destination.password)
        throw Error("Pair a tab on the destination origin before navigating");
      if (action.new_tab) throw Error("Open and pair the new tab first");
      usedGrants.set(grant.id, Date.parse(grant.expires_at));
      if (!sameTarget(await binding(), target)) throw Error("Browser target changed before dispatch");
      checkAlive();
      dispatched = true;
      await chrome.tabs.update(target.tab_id, { url: destination.href });
      checkAlive();
      await waitForNavigation(target.tab_id, destination.href, checkAlive);
      response.data = { browser_target: await binding() };
    } else throw Error("Unknown browser operation");
    checkAlive();
    response.success = true;
  } catch (error) { response.error = String(error.message || error).slice(0, 4000); }
  if (replyPort === port) replyPort.postMessage(response);
}

async function waitForNavigation(tabId, expected, checkAlive) {
  const end = Date.now() + 15000;
  while (Date.now() < end) {
    checkAlive();
    const tab = await chrome.tabs.get(tabId);
    checkAlive();
    if (tab.status === "complete") {
      if (tab.url !== expected) throw Error("Navigation redirected away from the approved URL");
      return;
    }
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw Error("Navigation timed out");
}

async function dom(target, operation, payload) {
  const results = await chrome.scripting.executeScript({ target: { tabId: target.tab_id, documentIds: [target.document_id] }, world: "ISOLATED", func: pageOperation, args: [target.origin, operation, payload] });
  if (results.length !== 1 || results[0].documentId !== target.document_id || !results[0].result) throw Error("Page did not provide a structured result");
  if (results[0].result.error) throw Error(results[0].result.error);
  return results[0].result;
}

export function pageOperation(origin, operation, payload) {
  try {
    if (location.origin !== origin) throw Error("Page origin changed");
    const safe = e => !e.matches('input[type="password"],input[autocomplete*="cc-"],input[autocomplete*="password"]')
      && !e.closest('[data-sage-private],[hidden],[aria-hidden="true"],script,style,noscript,template');
    const clip = (value, limit) => Array.from(String(value || "").replace(/\s+/g, " ").trim()).slice(0, limit).join("");
    const visible = e => e.getClientRects().length > 0;
    const roleFor = e => {
      const explicit = (e.getAttribute("role") || "").trim().split(/\s+/)[0];
      if (explicit) return explicit;
      const tag = e.tagName.toLowerCase();
      const type = (e.getAttribute("type") || "text").toLowerCase();
      if (tag === "button" || (tag === "input" && ["button", "submit", "reset", "image"].includes(type))) return "button";
      if (tag === "a" && e.getAttribute("href") !== null) return "link";
      if (tag === "input" && type === "checkbox") return "checkbox";
      if (tag === "input" && type === "radio") return "radio";
      if (tag === "input" && type === "range") return "slider";
      if (tag === "select") return "combobox";
      if (tag === "textarea" || (tag === "input" && !["hidden", "password", "checkbox", "radio", "range", "button", "submit", "reset", "image", "file"].includes(type))) return "textbox";
      if (e.isContentEditable || e.getAttribute("contenteditable") === "true") return "textbox";
      return "";
    };
    const kindFor = (e, role) => {
      const tag = e.tagName.toLowerCase();
      const type = (e.getAttribute("type") || "text").toLowerCase();
      if (role === "button") return "button";
      if (role === "link") return "link";
      if (role === "checkbox") return "checkbox";
      if (role === "switch") return "switch";
      if (role === "slider") return "slider";
      if (role === "radio") return "radio";
      if (role === "tab") return "tab";
      if (role === "menuitem" || role.startsWith("menuitem")) return "menuitem";
      if (role === "combobox" || role === "listbox" || tag === "select") return "select";
      if (role === "textbox" || role === "searchbox" || tag === "textarea" || (tag === "input" && !["hidden", "password", "checkbox", "radio", "range", "button", "submit", "reset", "image", "file"].includes(type))) return "textbox";
      return "";
    };
    const boundedText = root => {
      const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
      const parts = []; let size = 0; let nodes = 0;
      while (size < 240 && nodes < 32) {
        const node = walker.nextNode();
        if (!node) break;
        nodes++;
        const parent = node.parentElement;
        if (!parent || !safe(parent) || !visible(parent)) continue;
        const style = getComputedStyle(parent);
        if (style.display === "none" || style.visibility === "hidden") continue;
        const text = clip(node.nodeValue, 240 - size);
        if (!text) continue;
        parts.push(text); size += text.length + 1;
      }
      return clip(parts.join(" "), 160);
    };
    const labelFor = (e, role) => {
      const direct = clip(e.getAttribute("aria-label") || e.getAttribute("title"), 160);
      if (direct) return direct;
      const associated = e.labels?.[0];
      if (associated && safe(associated) && visible(associated)) return boundedText(associated);
      // Never inspect editable text: a contenteditable element may contain
      // what the user is currently typing.
      if (role === "textbox" || e.isContentEditable || e.getAttribute("contenteditable") === "true") return "";
      return boundedText(e);
    };
    const ancestorNames = e => {
      const names = [];
      let parent = e.parentElement;
      while (parent && names.length < 2) {
        if (!safe(parent)) break;
        const role = roleFor(parent);
        const landmark = ["main", "nav", "header", "footer", "aside", "section"].includes(parent.tagName.toLowerCase());
        const name = clip(parent.getAttribute("aria-label") || parent.getAttribute("title"), 64);
        if (name && (role || landmark)) names.push(name);
        parent = parent.parentElement;
      }
      return names.reverse();
    };
    const visiblePageText = () => {
      const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
      const parts = []; let size = 0; let nodes = 0;
      while (size < 8000 && nodes < 600) {
        const node = walker.nextNode();
        if (!node) break;
        nodes++;
        const parent = node.parentElement;
        if (!parent || !safe(parent) || !visible(parent)) continue;
        const style = getComputedStyle(parent);
        if (style.display === "none" || style.visibility === "hidden") continue;
        const text = String(node.nodeValue || "").replace(/\s+/g, " ").trim();
        if (!text) continue;
        const clipped = text.slice(0, Math.max(0, 8000 - size));
        parts.push(clipped); size += clipped.length + 1;
      }
      return parts.join("\n").slice(0, 8000);
    };
    const candidates = selector => {
      if (!selector || !Object.values(selector).some(v => typeof v === "string" && v.length)) throw Error("Explicit selector required");
      const source = selector.browser_selector ? document.querySelectorAll(selector.browser_selector) : document.querySelectorAll('button,input,textarea,a,select,[role],[contenteditable="true"]');
      if (source.length > 500) throw Error("Selector is too broad");
      return [...source].filter(e => visible(e) && safe(e) && (!selector.automation_id || e.id === selector.automation_id)
        && (!selector.label || label(e) === selector.label) && (!selector.role || (e.getAttribute("role") || e.tagName.toLowerCase()) === selector.role));
    };
    if (operation === "discover") {
      const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_ELEMENT);
      const controls = []; let visited = 0; let truncated = false;
      while (visited < 2048) {
        const e = walker.nextNode();
        if (!e) break;
        visited++;
        if (!safe(e) || !visible(e)) continue;
        const role = roleFor(e), kind = kindFor(e, role);
        if (!role || !kind) continue;
        const label = labelFor(e, role);
        const lowerLabel = label.toLowerCase();
        if (!label || label.includes("@") || label.includes("://") || lowerLabel.startsWith("www.")
          || Array.from(label).filter(character => /[0-9]/.test(character)).length >= 7
          || /\b(?:api[_ -]?key|authorization|bearer|password|passcode|secret|token)\s*[:=]\s*\S+/i.test(label)
          || /\b(?:sk-|ghp_|github_pat_|xox[bcoprs]-|AKIA)[A-Za-z0-9_-]{9,}/.test(label)
          || ["password", "passcode", "api key", "apikey", "authorization", "bearer", "secret", "token", "credit card", "card number"].some(term => lowerLabel.includes(term))) continue;
        const style = getComputedStyle(e);
        const enabled = !e.disabled && !e.readOnly && e.getAttribute("aria-disabled") !== "true"
          && style.display !== "none" && style.visibility !== "hidden";
        controls.push({ role: clip(role, 48), kind, label, enabled, ancestors: ancestorNames(e) });
        if (controls.length === 28) {
          truncated = !!walker.nextNode();
          break;
        }
      }
      if (visited === 2048) truncated = true;
      return { controls, truncated };
    }
    if (operation === "reference") {
      const selection = getSelection();
      let selectedText = "";
      if (selection && selection.rangeCount > 0) {
        const range = selection.getRangeAt(0);
        const anchors = [selection.anchorNode?.parentElement, selection.focusNode?.parentElement].filter(Boolean);
        const privateNodes = [...document.querySelectorAll('[data-sage-private]')].slice(0, 1000);
        const tooManyPrivateNodes = document.querySelectorAll('[data-sage-private]').length > 1000;
        const crossesPrivate = tooManyPrivateNodes || privateNodes.some(element => {
          try { return range.intersectsNode(element); } catch (_) { return false; }
        });
        if (!crossesPrivate && anchors.length > 0 && anchors.every(safe)) selectedText = String(selection).slice(0, 4000);
      }
      const pageText = payload.include_page_text ? visiblePageText() : "";
      return { title: document.title.slice(0, 300), url: location.href, selected_text: selectedText, page_text: pageText };
    }
    if (operation === "observe") return { present: candidates(payload.selector).length === 1 };
    throw Error("This page operation is not registered");
  } catch (error) { return { error: String(error.message || error).slice(0, 4000) }; }
}

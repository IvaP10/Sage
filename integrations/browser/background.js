// A user gesture pairs one tab. Every execution binds its prepared document.
let port, paired;
const usedGrants = new Map();
const generations = new Map();
let queue = Promise.resolve();

chrome.tabs.onUpdated.addListener((id, change) => {
  if (change.status === "loading" || change.url) generations.set(id, (generations.get(id) || 0) + 1);
});
chrome.tabs.onRemoved.addListener(id => {
  generations.delete(id);
  if (paired?.tabId === id) { port?.disconnect(); port = undefined; paired = undefined; }
});
chrome.action.onClicked.addListener(async tab => {
  port?.disconnect(); port = undefined; paired = undefined; usedGrants.clear();
  if (!Number.isInteger(tab.id) || !/^https?:/.test(tab.url || "")) return;
  generations.set(tab.id, (generations.get(tab.id) || 0) + 1);
  paired = { tabId: tab.id, windowId: tab.windowId, origin: new URL(tab.url).origin };
  const connection = chrome.runtime.connectNative("com.ivanpadeliya.sage.browser");
  port = connection;
  connection.onMessage.addListener(request => { queue = queue.then(() => handle(request, connection)).catch(() => {}); });
  connection.onDisconnect.addListener(() => {
    const message = chrome.runtime.lastError?.message;
    if (port !== connection) return;
    port = undefined; paired = undefined; usedGrants.clear();
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

async function handle(request, replyPort) {
  if (replyPort !== port) return;
  const response = { request_id: request.request_id, success: false, data: {} };
  try {
    if (!paired || !Number.isFinite(request.expires_at_unix_ms) || request.expires_at_unix_ms <= Date.now()) throw Error("Browser request is stale or unpaired");
    const target = await binding();
    const payload = request.payload || {};
    if (request.operation === "binding") response.data = target;
    else if (request.operation === "context") response.data = { ...await dom(target, "context", {}), browser_target: target };
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
      if (!sameTarget(await binding(), target) || replyPort !== port || request.expires_at_unix_ms <= Date.now()) throw Error("Browser target or request changed before dispatch");
      await chrome.tabs.update(target.tab_id, { url: destination.href });
      await waitForNavigation(target.tab_id, destination.href);
      response.data = { browser_target: await binding() };
    } else throw Error("Unknown browser operation");
    response.success = true;
  } catch (error) { response.error = String(error.message || error).slice(0, 4000); }
  if (replyPort === port) replyPort.postMessage(response);
}

async function waitForNavigation(tabId, expected) {
  const end = Date.now() + 15000;
  while (Date.now() < end) {
    const tab = await chrome.tabs.get(tabId);
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
    const safe = e => !e.matches('input[type="password"],input[autocomplete*="cc-"],input[autocomplete*="password"]') && !e.closest('[data-sage-private]');
    const label = e => (e.getAttribute("aria-label") || e.labels?.[0]?.textContent || e.textContent || "").trim().slice(0, 160);
    const visible = e => e.getClientRects().length > 0;
    const candidates = selector => {
      if (!selector || !Object.values(selector).some(v => typeof v === "string" && v.length)) throw Error("Explicit selector required");
      const source = selector.browser_selector ? document.querySelectorAll(selector.browser_selector) : document.querySelectorAll('button,input,textarea,a,select,[role],[contenteditable="true"]');
      if (source.length > 500) throw Error("Selector is too broad");
      return [...source].filter(e => visible(e) && safe(e) && (!selector.automation_id || e.id === selector.automation_id)
        && (!selector.label || label(e) === selector.label) && (!selector.role || (e.getAttribute("role") || e.tagName.toLowerCase()) === selector.role));
    };
    if (operation === "context") {
      const active = document.activeElement;
      return { title: document.title.slice(0, 500), selected_text: active && safe(active) ? String(getSelection() || "").slice(0, 2000) : "",
        elements: [...document.querySelectorAll('button,input,textarea,a,[role],[contenteditable="true"]')].filter(e => visible(e) && safe(e)).slice(0, 60)
          .map(e => ({ role: e.getAttribute("role") || e.tagName.toLowerCase(), label: label(e), automation_id: e.id || undefined })) };
    }
    if (operation === "observe") return { present: candidates(payload.selector).length === 1 };
    throw Error("This page operation is not registered");
  } catch (error) { return { error: String(error.message || error).slice(0, 4000) }; }
}

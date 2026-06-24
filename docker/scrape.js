#!/usr/bin/env node
// Headless-Chrome page fetcher for the hermes-eval sandbox.
//   node /opt/hermes-eval/scrape.js <url> [--selector <css>] [--html] [--timeout <ms>] [--wait <state>]
// Renders the page with playwright + stealth (so JS-heavy / lightly-bot-gated
// sites work, unlike a plain HTTP GET) and prints the result to stdout.
//   default      -> visible text content of the page
//   --selector S -> textContent of each element matching S, one per line
//   --html       -> full rendered outerHTML
// Pairs with ddgs: ddgs finds URLs, this fetches the rendered body.

const { chromium } = require("playwright-extra");
const stealth = require("puppeteer-extra-plugin-stealth")();
chromium.use(stealth);

function parseArgs(argv) {
  const a = { url: null, selector: null, html: false, timeout: 30000, wait: "networkidle" };
  const rest = argv.slice(2);
  for (let i = 0; i < rest.length; i++) {
    const t = rest[i];
    if (t === "--selector") a.selector = rest[++i];
    else if (t === "--html") a.html = true;
    else if (t === "--timeout") a.timeout = parseInt(rest[++i], 10) || 30000;
    else if (t === "--wait") a.wait = rest[++i];
    else if (!a.url) a.url = t;
  }
  return a;
}

(async () => {
  const args = parseArgs(process.argv);
  if (!args.url) {
    console.error("usage: node scrape.js <url> [--selector <css>] [--html] [--timeout <ms>] [--wait <load|domcontentloaded|networkidle>]");
    process.exit(2);
  }

  const browser = await chromium.launch({ headless: true, channel: "chromium-headless-shell" });
  try {
    const ctx = await browser.newContext({
      userAgent:
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
      viewport: { width: 1280, height: 800 },
    });
    const page = await ctx.newPage();
    await page.goto(args.url, { waitUntil: args.wait, timeout: args.timeout });

    if (args.html) {
      console.log(await page.content());
    } else if (args.selector) {
      const items = await page.$$eval(args.selector, (els) =>
        els.map((e) => (e.textContent || "").trim()).filter(Boolean)
      );
      console.log(items.join("\n"));
    } else {
      const text = await page.evaluate(() => document.body ? document.body.innerText : "");
      console.log(text);
    }
  } catch (e) {
    console.error("scrape error:", e.message.split("\n")[0]);
    process.exit(1);
  } finally {
    await browser.close();
  }
})();

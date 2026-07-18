"""
JARVIS Mark II — your trained model, now with internet access.

The tiny LLM you trained (checkpoints/jarvis.pt) handles personality and
small talk. Factual questions are routed to real sources:

    time / date      -> your Mac's clock
    weather          -> wttr.in            (free, no key)
    news headlines   -> BBC RSS feed       (free, no key)
    who is / what is -> Wikipedia          (free, no key)
    math             -> computed locally

Note: questions routed to weather/news/Wikipedia are sent to those public
services over the internet. Small talk never leaves your machine.

Run:  python3 jarvis.py
      python3 jarvis.py --voice     replies spoken aloud (macOS)
"""
import argparse
import ast
import datetime
import json
import operator
import os
import random
import re
import subprocess
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

UA = {"User-Agent": "jarvis-llm/1.0 (hobby assistant project)"}


def fetch(url, timeout=10):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


# ------------------------------------------------- the big brain (Llama)
# A real 1-billion-parameter open-weight model, running locally via Ollama.
# Used for conversation when available; the tools below still handle
# live facts, and your from-scratch model is the fallback personality.

OLLAMA = "http://localhost:11434"
LLAMA_MODEL = "llama3.2:1b"
SYSTEM_PROMPT = (
    "You are JARVIS, a witty British AI butler in the style of Iron Man's "
    "assistant. Address the user as 'sir'. Be helpful, dryly humorous, and "
    "concise: one to three sentences unless asked for more. Never break "
    "character or mention being a language model."
)


def post_json(url, payload, timeout=180):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def ollama_running():
    try:
        fetch(OLLAMA + "/api/version", timeout=2)
        return True
    except Exception:
        return False


def ollama_has_model():
    try:
        tags = json.loads(fetch(OLLAMA + "/api/tags", timeout=3))
        return any(m["name"].startswith(LLAMA_MODEL)
                   for m in tags.get("models", []))
    except Exception:
        return False


def ensure_ollama():
    """Return True if the Ollama server is up (starting it if we can)."""
    if ollama_running():
        return ollama_has_model()
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (os.path.join(here, "bin", "ollama"),
                      "/usr/local/bin/ollama", "/opt/homebrew/bin/ollama"):
        if os.path.exists(candidate):
            subprocess.Popen([candidate, "serve"],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            for _ in range(20):
                time.sleep(0.5)
                if ollama_running():
                    return ollama_has_model()
    return False


def llama_reply(messages):
    resp = post_json(OLLAMA + "/api/chat", {
        "model": LLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 250},
    })
    return resp["message"]["content"].strip()


# ------------------------------------------------------------------ tools

def tool_time(q):
    now = datetime.datetime.now()
    if re.search(r"\b(date|day)\b", q):
        return "Today is %s, %s %d, %d, sir." % (
            now.strftime("%A"), now.strftime("%B"), now.day, now.year)
    return "It is %s, sir." % now.strftime("%I:%M %p").lstrip("0")


def tool_weather(q):
    m = re.search(r"\b(?:in|at|for)\s+([a-zA-Z][a-zA-Z .-]*?)[?.!]*$", q)
    place = m.group(1).strip() if m else ""
    url = "https://wttr.in/%s?format=j1" % urllib.parse.quote(place)
    data = json.loads(fetch(url))
    cur = data["current_condition"][0]
    area = data["nearest_area"][0]["areaName"][0]["value"]
    desc = cur["weatherDesc"][0]["value"].lower()
    return random.choice([
        "My satellite uplink reports %(desc)s in %(area)s, sir. %(t)s degrees Celsius, feels like %(f)s, humidity at %(h)s percent.",
        "Live readings for %(area)s, sir: %(desc)s, %(t)s degrees Celsius (feels like %(f)s), humidity %(h)s percent.",
    ]) % {"desc": desc, "area": area, "t": cur["temp_C"],
          "f": cur["FeelsLikeC"], "h": cur["humidity"]}


def tool_news(q):
    xml_text = fetch("https://feeds.bbci.co.uk/news/rss.xml")
    root = ET.fromstring(xml_text)
    titles = [i.findtext("title") for i in root.iter("item")][:5]
    lines = "\n".join("  %d. %s" % (n + 1, t) for n, t in enumerate(titles))
    return "The latest headlines from the BBC, sir:\n%s" % lines


def tool_wiki(topic):
    topic = topic.strip().rstrip("?.!").strip()
    topic = re.sub(r"^(the|a|an)\s+", "", topic)
    if not topic:
        return None
    s = fetch("https://en.wikipedia.org/w/api.php?action=opensearch"
              "&limit=1&format=json&search=" + urllib.parse.quote(topic))
    hits = json.loads(s)[1]
    if not hits:
        return None
    page = json.loads(fetch(
        "https://en.wikipedia.org/api/rest_v1/page/summary/"
        + urllib.parse.quote(hits[0])))
    extract = page.get("extract", "")
    if not extract:
        return None
    # keep it to a couple of sentences
    if len(extract) > 450:
        cut = extract[:450]
        extract = cut[:cut.rfind(".") + 1] or cut
    prefix = random.choice([
        "Consulting the global archives, sir. ",
        "According to my records, sir: ",
        "The archives say, sir: ",
    ])
    return prefix + extract


MATH_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def calc(expr):
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in MATH_OPS:
            l, r = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(r) > 1000:
                raise ValueError("exponent too large")
            return MATH_OPS[type(n.op)](l, r)
        if isinstance(n, ast.UnaryOp) and type(n.op) in MATH_OPS:
            return MATH_OPS[type(n.op)](ev(n.operand))
        raise ValueError("not arithmetic")
    return ev(ast.parse(expr, mode="eval"))


def tool_math(q):
    m = re.search(r"[-(]*\d[\d\s.()+\-*/x^%]*", q)
    if not m:
        return None
    expr = m.group(0).replace("x", "*").replace("^", "**").strip()
    if not re.search(r"[+\-*/%]", expr.lstrip("-(")):
        return None  # a lone number, not a calculation
    try:
        val = calc(expr)
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
        return None
    if isinstance(val, float) and val.is_integer():
        val = int(val)
    elif isinstance(val, float):
        val = round(val, 6)
    return "That comes to %s, sir. Computed locally, naturally." % val


KNOW_RE = re.compile(
    r"^(?:who is|who was|who are|what is|what's|whats|what are|what was|"
    r"where is|tell me about|search for|search|look up|wiki|define|explain)"
    r"\s+(.+)$")


def route(q):
    """Decide whether a message needs a real-world tool.
    Returns a reply string, or None to let the LLM handle it."""
    q = q.lower().strip()
    if re.search(r"\d\s*[+\-*/x^%]\s*\d", q):
        ans = tool_math(q)
        if ans:
            return ans
    if re.search(r"\b(weather|rain|umbrella|forecast|temperature|sunny|snow)\b", q):
        return tool_weather(q)
    if re.search(r"\btime\b", q) or re.search(r"\b(what day|date)\b", q):
        return tool_time(q)
    if re.search(r"\b(news|headline|headlines)\b", q):
        return tool_news(q)
    m = KNOW_RE.match(q)
    if m and not re.search(r"\b(you|your|yourself)\b", q):
        return tool_wiki(m.group(1))
    return None


# ------------------------------------------------------- the personality

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/jarvis.pt")
    p.add_argument("--voice", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--brain", choices=["auto", "llama", "tiny"], default="llama",
                   help="auto: Llama 1B if available, else your tiny model")
    args = p.parse_args()

    tiny_loaded = False
    device = None
    model = None
    config = None
    stoi = None
    itos = None

    def ensure_tiny():
        nonlocal tiny_loaded, device, model, config, stoi, itos
        if tiny_loaded:
            return
        import torch
        from model import GPT, GPTConfig

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        ckpt = torch.load(args.ckpt, map_location=device)
        config = GPTConfig(**ckpt["config"])
        model = GPT(config).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        stoi, itos = ckpt["stoi"], ckpt["itos"]
        tiny_loaded = True

    def encode(s):
        ensure_tiny()
        return [stoi[c] for c in s if c in stoi]

    def decode(ids):
        ensure_tiny()
        return "".join(itos[i] for i in ids)

    def llm_reply(history):
        ensure_tiny()
        import torch
        context = history[-config.block_size:]
        idx = torch.tensor([encode(context)], device=device)

        def stop_check(new_ids):
            tail = decode(new_ids[-8:])
            return "\nUSER" in tail or tail.endswith("\n\n")

        _, new_ids = model.generate(
            idx, max_new_tokens=220,
            temperature=args.temperature, top_k=40, stop_check=stop_check)
        reply = decode(new_ids)
        for cutmark in ("\nUSER", "\n\n"):
            if cutmark in reply:
                reply = reply.split(cutmark)[0]
        return reply.strip()

    use_llama = args.brain != "tiny" and ensure_ollama()
    if args.brain == "llama" and not use_llama:
        print("Llama is not available (is the model pulled?). Falling back "
              "to your from-scratch model.\n")

    brain_name = "Llama 3.2 1B" if use_llama else "from-scratch tiny model"
    print("JARVIS Mark III online, sir. Brain: %s + live uplink. "
          "(ctrl+c to exit)\n" % brain_name)

    history = ""  # context for the tiny model
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]  # for Llama
    while True:
        try:
            user = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nJARVIS: Powering down, sir.")
            break
        if not user:
            continue

        try:
            reply = route(user)
        except Exception:
            reply = ("I am afraid my uplink to the internet has failed, sir. "
                     "Do check the connection.")

        if reply is not None:
            source = "live"
        elif use_llama:
            source = "llama 1B"
            try:
                # system prompt + the last few turns + the new message
                ctx = ([messages[0]] + messages[1:][-10:]
                       + [{"role": "user", "content": user}])
                reply = llama_reply(ctx)
            except Exception:
                source = "tiny model"
                reply = None  # fall through to the tiny model below

        if reply is None:
            source = "tiny model"
            history += "USER: %s\nJARVIS:" % user.lower()
            reply = llm_reply(history)
            history += " %s\n" % reply
        else:
            history += "USER: %s\nJARVIS: %s\n" % (user.lower(), reply)

        # record the turn once, whichever brain answered, so context flows
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": reply})

        print("JARVIS: %s   [%s]\n" % (reply, source))

        if args.voice and reply:
            subprocess.run(["say", "-v", "Daniel", reply],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()

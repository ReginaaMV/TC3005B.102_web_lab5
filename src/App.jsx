import { useEffect, useRef, useState } from "react";
import { chunkText, embedTexts, retrieve, generateAnswer } from "./rag.js";

export default function App() {
  const [apiKey, setApiKey] = useState(() => localStorage.getItem("gemini_key") || "");
  const [docs, setDocs] = useState([]); // { name, text }
  const [pasted, setPasted] = useState("");
  const [index, setIndex] = useState([]); // { source, text, vector }
  const [indexing, setIndexing] = useState(false);
  const [messages, setMessages] = useState([]); // { role, text, sources? }
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const endRef = useRef(null);

  useEffect(() => {
    localStorage.setItem("gemini_key", apiKey);
  }, [apiKey]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function handleFiles(e) {
    const files = Array.from(e.target.files);
    const loaded = await Promise.all(
      files.map(async (f) => ({ name: f.name, text: await f.text() }))
    );
    setDocs((d) => [...d, ...loaded]);
    e.target.value = "";
  }

  function addPasted() {
    if (!pasted.trim()) return;
    setDocs((d) => [...d, { name: `Texto pegado ${d.length + 1}`, text: pasted }]);
    setPasted("");
  }

  async function buildIndex() {
    setError("");
    if (!apiKey) return setError("Escribe tu API key de Gemini.");
    if (!docs.length) return setError("Agrega al menos un documento.");
    setIndexing(true);
    try {
      const pieces = docs.flatMap((d) =>
        chunkText(d.text).map((text) => ({ source: d.name, text }))
      );
      const vectors = await embedTexts(apiKey, pieces.map((p) => p.text), "RETRIEVAL_DOCUMENT");
      setIndex(pieces.map((p, i) => ({ ...p, vector: vectors[i] })));
      setMessages([]);
    } catch (err) {
      setError(err.message);
    } finally {
      setIndexing(false);
    }
  }

  async function send(e) {
    e.preventDefault();
    const question = input.trim();
    if (!question || loading) return;
    if (!index.length) return setError("Indexa tus documentos antes de preguntar.");
    setError("");
    setInput("");
    const history = messages.map(({ role, text }) => ({ role, text }));
    setMessages((m) => [...m, { role: "user", text: question }]);
    setLoading(true);
    try {
      const sources = await retrieve(apiKey, question, index);
      const answer = await generateAnswer(apiKey, question, sources, history);
      setMessages((m) => [...m, { role: "model", text: answer, sources }]);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <aside className="panel">
        <h1>Chat con tus documentos</h1>
        <p className="muted">Gemini responde usando solo el contenido que cargues.</p>

        <label htmlFor="key">API key de Gemini</label>
        <input
          id="key"
          type="password"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          placeholder="Pega tu API key"
        />
        <p className="hint">Se guarda solo en tu navegador.</p>

        <label htmlFor="files">Documentos (.txt, .md)</label>
        <input id="files" type="file" multiple accept=".txt,.md,text/plain" onChange={handleFiles} />

        <label htmlFor="paste">O pega texto</label>
        <textarea
          id="paste"
          rows={4}
          value={pasted}
          onChange={(e) => setPasted(e.target.value)}
          placeholder="Pega aquí apuntes, políticas, artículos…"
        />
        <button className="secondary" onClick={addPasted} disabled={!pasted.trim()}>
          Agregar texto
        </button>

        {docs.length > 0 && (
          <ul className="docs">
            {docs.map((d, i) => (
              <li key={i}>
                <span>{d.name}</span>
                <button
                  className="link"
                  onClick={() => setDocs((x) => x.filter((_, j) => j !== i))}
                  aria-label={`Quitar ${d.name}`}
                >
                  Quitar
                </button>
              </li>
            ))}
          </ul>
        )}

        <button onClick={buildIndex} disabled={indexing || !docs.length}>
          {indexing ? "Indexando…" : "Indexar documentos"}
        </button>
        {index.length > 0 && <p className="ok">{index.length} fragmentos listos para consultar.</p>}
      </aside>

      <main className="chat">
        <div className="messages" aria-live="polite">
          {messages.length === 0 && (
            <p className="empty">
              Carga documentos, indéxalos y haz tu primera pregunta.
            </p>
          )}
          {messages.map((m, i) => (
            <div key={i} className={`msg ${m.role}`}>
              <p>{m.text}</p>
              {m.sources && (
                <details>
                  <summary>Fuentes consultadas ({m.sources.length})</summary>
                  {m.sources.map((s, j) => (
                    <blockquote key={j}>
                      <strong>
                        {s.source} · similitud {s.score.toFixed(2)}
                      </strong>
                      <br />
                      {s.text.slice(0, 280)}
                      {s.text.length > 280 ? "…" : ""}
                    </blockquote>
                  ))}
                </details>
              )}
            </div>
          ))}
          {loading && <div className="msg model typing">Buscando y redactando…</div>}
          <div ref={endRef} />
        </div>

        {error && <p className="error" role="alert">{error}</p>}

        <form onSubmit={send} className="composer">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Pregunta algo sobre tus documentos"
            aria-label="Pregunta"
          />
          <button type="submit" disabled={loading || !input.trim()}>
            Enviar
          </button>
        </form>
      </main>
    </div>
  );
}

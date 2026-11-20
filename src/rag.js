// Lógica RAG: chunking -> embeddings -> búsqueda por similitud -> generación.
const BASE = "https://generativelanguage.googleapis.com/v1beta";
export const EMBED_MODEL = "gemini-embedding-001";
export const CHAT_MODEL = "gemini-2.5-flash";

async function callGemini(apiKey, path, body) {
  const res = await fetch(`${BASE}/${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "x-goog-api-key": apiKey },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data?.error?.message || `Error ${res.status} al llamar a la API`);
  }
  return data;
}

// 1. Divide el texto en fragmentos con traslape.
export function chunkText(text, size = 800, overlap = 150) {
  const clean = text.replace(/\r/g, "").trim();
  const chunks = [];
  let start = 0;
  while (start < clean.length) {
    let end = Math.min(start + size, clean.length);
    if (end < clean.length) {
      const cut = clean.lastIndexOf("\n", end);
      const dot = clean.lastIndexOf(". ", end);
      const best = Math.max(cut, dot);
      if (best > start + size / 2) end = best + 1;
    }
    const piece = clean.slice(start, end).trim();
    if (piece) chunks.push(piece);
    if (end >= clean.length) break;
    start = end - overlap;
  }
  return chunks;
}

// 2. Convierte textos en vectores (en lotes de 100).
export async function embedTexts(apiKey, texts, taskType) {
  const vectors = [];
  for (let i = 0; i < texts.length; i += 100) {
    const batch = texts.slice(i, i + 100);
    const data = await callGemini(apiKey, `models/${EMBED_MODEL}:batchEmbedContents`, {
      requests: batch.map((t) => ({
        model: `models/${EMBED_MODEL}`,
        content: { parts: [{ text: t }] },
        taskType,
      })),
    });
    vectors.push(...data.embeddings.map((e) => e.values));
  }
  return vectors;
}

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

// 3. Recupera los fragmentos más parecidos a la pregunta.
export async function retrieve(apiKey, question, index, k = 4) {
  const [qVec] = await embedTexts(apiKey, [question], "RETRIEVAL_QUERY");
  return index
    .map((item) => ({ ...item, score: cosine(qVec, item.vector) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}

// 4. Genera la respuesta usando solo el contexto recuperado.
export async function generateAnswer(apiKey, question, sources, history) {
  const context = sources
    .map((s, i) => `[Fragmento ${i + 1} - ${s.source}]\n${s.text}`)
    .join("\n\n");

  const data = await callGemini(apiKey, `models/${CHAT_MODEL}:generateContent`, {
    systemInstruction: {
      parts: [{
        text:
          "Eres un asistente que responde en español usando únicamente el contexto proporcionado. " +
          "Si la respuesta no está en el contexto, dilo claramente en lugar de inventarla. " +
          "Cita el número de fragmento cuando sea útil.",
      }],
    },
    contents: [
      ...history.map((m) => ({
        role: m.role === "user" ? "user" : "model",
        parts: [{ text: m.text }],
      })),
      { role: "user", parts: [{ text: `Contexto:\n${context}\n\nPregunta: ${question}` }] },
    ],
  });

  return data.candidates?.[0]?.content?.parts?.map((p) => p.text).join("") || "Sin respuesta.";
}

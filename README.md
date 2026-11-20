# Chat RAG con Gemini (React + Vite)

App en React que se conecta a la API de Gemini para chatear con tus propios documentos (RAG).

## Cómo funciona
1. **Chunking**: el texto se divide en fragmentos de ~800 caracteres con traslape.
2. **Embeddings**: cada fragmento se vectoriza con `gemini-embedding-001`.
3. **Retrieval**: la pregunta se vectoriza y se buscan los 4 fragmentos con mayor similitud coseno.
4. **Generación**: `gemini-2.5-flash` responde usando solo ese contexto y se muestran las fuentes.

## Ejecutar
```bash
npm install
npm run dev
```
Abre http://localhost:5173, pega tu API key de [Google AI Studio](https://aistudio.google.com/app/apikey), carga documentos `.txt`/`.md` (o pega texto), pulsa **Indexar documentos** y pregunta.

> La API key se guarda solo en el `localStorage` de tu navegador. No la subas al repositorio.


const API_BASE_URL = "http://127.0.0.1:8000";

const pdfInput = document.getElementById("pdfFile");
const documentList = document.getElementById("documentList");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");
const chatArea = document.getElementById("chatArea");
const statusBox = document.getElementById("status");
const sendButton = document.getElementById("sendButton");
const welcome = document.getElementById("welcome");

let isIndexed = false;
let isBusy = false;

function setStatus(message) {
  statusBox.textContent = message;
}

function addMessage(role, text) {
  if (welcome) {
    welcome.remove();
  }

  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  const roleLabel = document.createElement("div");
  roleLabel.className = "message-role";
  roleLabel.textContent = role === "user" ? "YOU" : "KNOWLEDGE AI";

  const body = document.createElement("div");
  body.className = "message-body";
  body.textContent = text;

  wrapper.append(roleLabel, body);
  chatArea.appendChild(wrapper);

  return wrapper;
}

function addSources(messageElement, sources) {
  if (!Array.isArray(sources) || sources.length === 0) {
    return;
  }

  const container = document.createElement("div");
  container.className = "sources";

  const heading = document.createElement("h3");
  heading.textContent = `Sources (${sources.length})`;
  container.appendChild(heading);

  for (const source of sources) {
    const item = document.createElement("div");
    item.className = "source-item";

    const title = document.createElement("strong");
    title.textContent =
      `${source.filename || "Document"} — Page ${source.page ?? "?"}`;

    const snippet = document.createElement("p");
    snippet.textContent = source.snippet || "";

    item.append(title, snippet);
    container.appendChild(item);
  }

  messageElement.appendChild(container);
}

function addDocument(filename, pageCount, chunkCount) {
  if (documentList.querySelector(".muted")) {
    documentList.replaceChildren();
  }

  const item = document.createElement("div");
  item.className = "document-item";

  const title = document.createElement("strong");
  title.textContent = filename;

  const details = document.createElement("small");
  details.textContent =
    `${pageCount} pages · ${chunkCount} chunks indexed`;

  item.append(title, details);
  documentList.appendChild(item);
}

async function uploadPDF(file) {
  if (!file) return;

  isIndexed = false;
  pdfInput.disabled = true;
  sendButton.disabled = true;
  setStatus("Uploading PDF and creating embeddings...");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch(
      `${API_BASE_URL}/api/pdf/upload`,
      {
        method: "POST",
        body: formData,
      }
    );

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "PDF upload failed.");
    }

    if (data.is_likely_scanned || !data.chunks_indexed) {
      setStatus(data.message || "No searchable text found.");
      return;
    }

    isIndexed = true;

    addDocument(
      data.filename,
      data.page_count,
      data.chunks_indexed
    );

    setStatus(
      `Ready to chat with ${data.filename}.`
    );

    addMessage(
      "assistant",
      `Your PDF is ready! I indexed ${data.chunks_indexed} chunks. Ask me a question about the document.`
    );

  } catch (error) {
    setStatus(`Upload failed: ${error.message}`);
  } finally {
    pdfInput.disabled = false;
    sendButton.disabled = !isIndexed;
    pdfInput.value = "";
  }
}

async function askQuestion(question) {
  if (!isIndexed) {
    setStatus("Please upload a PDF before asking questions.");
    return;
  }

  if (isBusy || !question.trim()) return;

  isBusy = true;
  sendButton.disabled = true;
  questionInput.disabled = true;

  addMessage("user", question);

  const answerElement = addMessage(
    "assistant",
    "Thinking... searching your document."
  );

  setStatus("Finding relevant passages and generating an answer...");

  try {
    const response = await fetch(
      `${API_BASE_URL}/api/chat`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question }),
      }
    );

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Chat request failed.");
    }

    const body = answerElement.querySelector(".message-body");
    body.textContent = data.answer || "No answer was returned.";

    addSources(answerElement, data.sources);

    setStatus("Answer generated.");

  } catch (error) {
    const body = answerElement.querySelector(".message-body");
    body.textContent = `Error: ${error.message}`;
    setStatus("Could not get an answer. Check the backend.");
  } finally {
    isBusy = false;
    sendButton.disabled = !isIndexed;
    questionInput.disabled = false;
    questionInput.focus();
    chatArea.scrollTop = chatArea.scrollHeight;
  }
}

pdfInput.addEventListener("change", () => {
  const file = pdfInput.files[0];

  if (file && file.size > 15 * 1024 * 1024) {
    setStatus("Please choose a PDF smaller than 15 MB.");
    pdfInput.value = "";
    return;
  }

  if (file && file.type && file.type !== "application/pdf") {
    setStatus("Please select a valid PDF file.");
    pdfInput.value = "";
    return;
  }

  uploadPDF(file);
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const question = questionInput.value.trim();

  if (!question) return;

  questionInput.value = "";
  await askQuestion(question);
});

document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => {
    const question = button.dataset.question;
    askQuestion(question);
  });
});

sendButton.disabled = true;

fetch(`${API_BASE_URL}/health`)
  .then((response) => {
    if (!response.ok) {
      throw new Error("Backend is unavailable.");
    }
    return response.json();
  })
  .then(() => setStatus("Backend connected. Upload a PDF to begin."))
  .catch(() => setStatus("Backend offline. Start the FastAPI server."));

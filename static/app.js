let sessionId = null;

const uploadForm = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const uploadStatus = document.getElementById("uploadStatus");
const askForm = document.getElementById("askForm");
const questionInput = document.getElementById("questionInput");
const answerMode = document.getElementById("answerMode");
const chatBox = document.getElementById("chatBox");
const summaryForm = document.getElementById("summaryForm");
const summaryTopic = document.getElementById("summaryTopic");
const summaryOutput = document.getElementById("summaryOutput");
const quizForm = document.getElementById("quizForm");
const quizTopic = document.getElementById("quizTopic");
const quizCount = document.getElementById("quizCount");
const quizOutput = document.getElementById("quizOutput");

uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = fileInput.files[0];
  if (!file) return;

  uploadStatus.textContent = "Uploading and indexing...";
  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/upload", { method: "POST", body: formData });
    const data = await readJsonSafe(res);
    if (!res.ok) throw new Error(data.detail || "Upload failed");
    uploadStatus.textContent = `Indexed: ${data.data.file_name} (${data.data.chunks_indexed} chunks)`;
  } catch (err) {
    uploadStatus.textContent = `Error: ${err.message}`;
  }
});

askForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  appendMessage("user", question);
  questionInput.value = "";

  const formData = new FormData();
  formData.append("question", question);
  formData.append("answer_mode", answerMode.value);
  if (sessionId) formData.append("session_id", sessionId);

  try {
    const res = await fetch("/ask", { method: "POST", body: formData });
    const data = await readJsonSafe(res);
    if (!res.ok) throw new Error(data.detail || "Question failed");
    sessionId = data.session_id;
    appendMessage("assistant", data.data.answer, data.data.sources);
  } catch (err) {
    appendMessage("assistant", `Error: ${err.message}`);
  }
});

summaryForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const topic = summaryTopic.value.trim();
  if (!topic) return;
  summaryOutput.textContent = "Generating summary...";

  const formData = new FormData();
  formData.append("topic", topic);
  const res = await fetch("/summarize", { method: "POST", body: formData });
  const data = await readJsonSafe(res);
  summaryOutput.textContent = data?.data?.summary || data.detail || "Failed.";
});

quizForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const topic = quizTopic.value.trim();
  if (!topic) return;
  quizOutput.textContent = "Generating quiz...";

  const formData = new FormData();
  formData.append("topic", topic);
  formData.append("count", quizCount.value || "5");
  const res = await fetch("/quiz", { method: "POST", body: formData });
  const data = await readJsonSafe(res);
  quizOutput.textContent = data?.data?.quiz || data.detail || "Failed.";
});

async function readJsonSafe(res) {
  const text = await res.text();
  try {
    return JSON.parse(text);
  } catch {
    return { detail: text || "Unexpected server response" };
  }
}

function appendMessage(role, text, sources = []) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = `${role === "user" ? "You" : "Assistant"}: ${text}`;

  if (role === "assistant" && sources?.length) {
    const s = document.createElement("div");
    s.className = "sources";
    s.textContent =
      "Sources: " +
      sources
        .map((x) => `${x.file_name} (page ${x.page >= 0 ? x.page : "N/A"}, chunk ${x.chunk_index})`)
        .join(" | ");
    div.appendChild(s);
  }
  chatBox.prepend(div);
}

const chatButton = document.getElementById('chat-widget-button');
const chatContainer = document.getElementById('chat-widget-container');
const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendButton = document.querySelector('.send-button');
const loader = document.getElementById('typing');
const graphContainer = document.getElementById('graph-container');
const maximizeButton = document.querySelector('.maximize-button');

// API config (IMPORTANT: Do NOT expose this in production)
const OPENROUTER_API_KEY = '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';
const OPENROUTER_API_URL = 'https://openrouter.ai/api/v1/chat/completions';

let isOpen = false;
let isMaximized = false;
let chatHistory = [];
let currentQuery = null;
let cy = null;
let currentEntities = null;

const systemInstruction = `
You are MOSDAC Help Assistant, an AI designed to assist users with:
- Satellite data products and services (INSAT-3D, INSAT-3DR, SCATSAT-1, etc.)
- Weather and oceanographic data interpretation
- Visualization tools, formats, and technical support for MOSDAC platforms.
Provide accurate and technical responses. If unsure, say "I'm not certain about that based on current data."
`;

// Welcome message
window.onload = () => {
  addMessage({
    role: 'assistant',
    content: 'Welcome to MOSDAC Help Assistant. How can I assist you with satellite data, weather information, or oceanographic services today?'
  });
};

// Toggle Chat
function toggleChat() {
  isOpen = !isOpen;
  chatContainer.classList.toggle('active');
  chatButton.style.display = isOpen ? 'none' : 'flex';

  if (isOpen) userInput.focus();
  if (!isOpen && isMaximized) toggleMaximize();
}

// Toggle Maximize
function toggleMaximize() {
  isMaximized = !isMaximized;
  chatContainer.classList.toggle('maximized');
  maximizeButton.querySelector('span').textContent = isMaximized ? '⛶' : '⛶';

  if (graphContainer.style.display === 'block' && cy) {
    setTimeout(() => {
      cy.resize();
      cy.fit();
    }, 300);
  }
}

// Format content
function formatMessageContent(content, isAssistant = false) {
  if (!isAssistant) return content;
  content = content.replace(/```(\w+)?\n([\s\S]*?)```/g, (_, lang, code) =>
    `<pre><code class="${lang || ''}">${code.trim()}</code></pre>`
  );
  content = content.replace(/`([^`]+)`/g, '<code>$1</code>');
  content = content.replace(/^\s*[-*]\s+(.+)$/gm, '<li>$1</li>');
  content = content.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
  content = content.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
  return content;
}

// Add message
function addMessage(message) {
  const messageDiv = document.createElement('div');
  messageDiv.className = `message ${message.role === 'user' ? 'user-message' : 'bot-message'}`;
  messageDiv.innerHTML = formatMessageContent(message.content, message.role === 'assistant');
  chatBox.appendChild(messageDiv);
  chatBox.scrollTop = chatBox.scrollHeight;
  chatHistory.push(message);
}

// Typing indicator
function toggleTyping(show) {
  loader.style.display = show ? 'flex' : 'none';
}

// Process input
async function processUserInput() {
  const message = userInput.value.trim();
  if (!message) return;

  userInput.value = '';
  currentQuery = message;

  addMessage({ role: 'user', content: message });
  toggleTyping(true);

  try {
    const response = await callOpenRouter(message);
    if (response) {
      addMessage({ role: 'assistant', content: response });
      const entities = extractEntities(response);
      if (entities.length > 0) currentEntities = entities;
    }
  } catch (error) {
    console.error('Chat error:', error);
    addMessage({
      role: 'assistant',
      content: 'Sorry, I had trouble understanding that. Please try again.'
    });
  } finally {
    toggleTyping(false);
  }
}

async function callOpenRouter(userMessage) {
  try {
    const messages = [];

    const fewShotExamples = `
Example:
User: What is INSAT-3DR?
Assistant: INSAT-3DR is an advanced meteorological satellite developed by ISRO used for weather forecasting and disaster warnings.

User: How can I access oceanographic data?
Assistant: You can access oceanographic data such as sea surface temperature and wind vectors on the MOSDAC portal (https://mosdac.gov.in) under the "Oceanography" section.
`;

    if (chatHistory.length === 0) {
      messages.push({
        role: 'user',
        content: `${systemInstruction}\n\n${fewShotExamples}\n\nUser: ${userMessage}`
      });
    } else {
      const trimmedHistory = chatHistory.slice(-6);
      messages.push(...trimmedHistory);
      messages.push({ role: 'user', content: userMessage });
    }

    const response = await fetch(OPENROUTER_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
        'HTTP-Referer': 'https://mosdac.gov.in',
        'X-Title': 'MOSDAC Help Assistant'
      },
      body: JSON.stringify({
        model: 'mistralai/mixtral-8x7b-instruct',  // switch from Claude
        messages,
        temperature: 0.4,
        max_tokens: 1000,
        stream: false
      })
      
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`API Error: ${response.status} ${response.statusText}\n${errorText}`);
      throw new Error(`OpenRouter API error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    console.log('Claude API response:', data);

    if (!data.choices || !data.choices[0]?.message?.content) {
      throw new Error('Invalid response format from Claude.');
    }

    return data.choices[0].message.content;
  } catch (err) {
    console.error('Claude API error:', err);

    // Display the error message in the chat
    addMessage({
      role: 'assistant',
      content: `⚠️ Something went wrong while processing your request. Error: ${err.message}`
    });

    return null;
  }
}


// Entity extraction
function extractEntities(text) {
  const entities = [];
  const keywords = [
    'INSAT-3D', 'INSAT-3DR', 'SCATSAT-1', 'OCEANSAT', 'SARAL',
    'satellite', 'data', 'weather', 'ocean', 'temperature',
    'rainfall', 'wind', 'humidity', 'pressure', 'cloud cover',
    'visualization', 'forecast', 'meteorological', 'oceanographic',
    'atmospheric', 'precipitation', 'radiation', 'sea surface'
  ];

  keywords.forEach(keyword => {
    if (text.toLowerCase().includes(keyword.toLowerCase())) {
      entities.push({
        id: keyword.toLowerCase().replace(/[^a-z0-9]/g, '_'),
        label: keyword
      });
    }
  });

  return entities;
}

// Update graph
function updateGraph() {
  if (!cy) {
    cy = cytoscape({
      container: document.getElementById('cy'),
      style: [
        {
          selector: 'node',
          style: {
            'background-color': '#0b3d91',
            'label': 'data(label)',
            'color': '#fff',
            'font-size': '12px',
            'text-valign': 'center',
            'text-halign': 'center',
            'width': 50,
            'height': 50
          }
        },
        {
          selector: 'edge',
          style: {
            'width': 2,
            'line-color': '#105bd8',
            'curve-style': 'bezier',
            'opacity': 0.7
          }
        }
      ]
    });
  }

  cy.elements().remove();

  currentEntities.forEach(entity => {
    cy.add({ group: 'nodes', data: { id: entity.id, label: entity.label } });
  });

  for (let i = 0; i < currentEntities.length; i++) {
    for (let j = i + 1; j < currentEntities.length; j++) {
      if (Math.random() < 0.3) {
        cy.add({
          group: 'edges',
          data: {
            id: `${currentEntities[i].id}-${currentEntities[j].id}`,
            source: currentEntities[i].id,
            target: currentEntities[j].id
          }
        });
      }
    }
  }

  cy.layout({
    name: 'cose',
    animate: true,
    randomize: true,
    componentSpacing: 50,
    nodeRepulsion: 400000,
    edgeElasticity: 100,
    nestingFactor: 5,
    gravity: 80,
    numIter: 1000,
    initialTemp: 200,
    coolingFactor: 0.95,
    minTemp: 1.0
  }).run();

  cy.fit();
}

// Graph visibility toggle
function toggleGraph() {
  const isVisible = graphContainer.style.display === 'block';
  graphContainer.style.display = isVisible ? 'none' : 'block';
  if (!isVisible && currentEntities) updateGraph();
}

// Open Navigation Tree
function openNavTree() {
  window.open('navigation.html', '_blank');
}

// Event listeners
userInput.addEventListener('keypress', (e) => {
  if (e.key === 'Enter') processUserInput();
});
sendButton.addEventListener('click', processUserInput);

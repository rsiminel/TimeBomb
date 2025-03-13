document.addEventListener("DOMContentLoaded", () => {
// ----- Global Game State -----
let players = [];
let handSize = 5;
let currentRound = 1;
let activeWires;
let numWires;
let numBomb = 1;
let posEvil = [];
let declarations = [];
let revealed = [];
let found = [];
let probabilitiesList = [[]];
let prior = [];


// ----- Routes -----
const backend = 'http://127.0.0.1:5000';

async function Declaration(probsList, decls, handSize, activeWires, posEvil, numBomb) {
  let response = await fetch(`${backend}/declaration`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      probs_list: probsList,
      decls: decls,
      hand_size: handSize,
      active_wires: activeWires,
      pos_evil: posEvil,
      num_bomb: numBomb
    })
  })
  .catch(error => console.error("Fetch Error:", error));
  return await response.json();
}

async function Cut(probsList, prior, decls, revealed, found, handSize, activeWires, posEvil, numBomb) {
  let response = await fetch(`${backend}/cut`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      probs_list: probsList,
      prior: prior,
      decls: decls,
      revealed: revealed,
      found: found,
      hand_size: handSize,
      active_wires: activeWires,
      pos_evil: posEvil,
      num_bomb: numBomb
    })
  })
  .catch(error => console.error("Fetch Error:", error));
  return await response.json();
}


// ----- Useful Functions -----
function updateProbabilitiesTable(wire, bomb, evil, score) {
  const createRow = (player, wire, bomb, evil, score) => `
    <tr>
      <td>${player}</td>
      <td>${(wire * 100).toFixed(1)}</td>
      <td>${(bomb * 100).toFixed(1)}</td>
      <td>${(evil * 100).toFixed(1)}</td>
      <td>${score.toFixed(3)}</td>
    </tr>
  `;
  const avg = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
  let tableHTML = `
    <table>
      <tr>
        <th>Player</th>
        <th>P_wire (%)</th>
        <th>P_bomb (%)</th>
        <th>P_evil (%)</th>
        <th>Expected Score</th>
      </tr>
      ${players.map((player, i) => createRow(player, wire[i], bomb[i], evil[i], score[i])).join('')}
      ${createRow('Average', avg(wire), avg(bomb), avg(evil), avg(score))}
    </table>
  `;
  document.getElementById("probabilitiesTable").innerHTML = tableHTML;
}

function log(message) {
  let logArea = document.getElementById("Sidebar").querySelector("code");
  logArea.textContent += message + "\n";
  logArea.scrollTop = logArea.scrollHeight;
}


// ----- Event Handlers -----
document.getElementById("startGame").addEventListener("click", () => {
  let playersInput = document.getElementById("playersInput").value;
  if (!playersInput) {
    alert("Please enter player names separated by commas.");
    return;
  }
  handSize = parseInt(document.getElementById("handSizeInput").value);
  if (isNaN(handSize) || handSize < 2) {
    alert("Invalid hand size.");
    return;
  }
  // Set posEvil configuration based on number of players
  players = playersInput.split(",").map(s => s.trim()).filter(s => s);
  if (players.length < 4) {
    alert("Not enough players (minimum 4).");
    return;
  } else if (players.length === 4) {
    posEvil = [[1, 2/5], [2, 3/5]];
  } else if (players.length < 7) {
    posEvil = [[2, 1.0]];
  } else if (players.length === 7) {
    posEvil = [[2, 3/8], [3, 5/8]];
  } else if (players.length === 8) {
    posEvil = [[3, 1.0]];
  } else {
    alert("Too many players (maximum 8).");
    return;
  }
  // Initialize game state
  currentRound = 1;
  activeWires = players.length;
  numWires = players.length * handSize;
  declarations = new Array(players.length).fill(0);
  revealed = new Array(players.length).fill(0);
  found = new Array(players.length).fill(0);
  probabilitiesList = Array.from({ length: posEvil.length }, () => []);
  prior = [];
  // Create declaration inputs for each player
  let declarationsDiv = document.getElementById("declarationsInputs");
  declarationsDiv.innerHTML = players.map((player, index) => `
    <label>${player}'s declaration:</label>
    <input type="number" min="0" value="0" id="decl_${index}">
    <br>
  `).join('');
  // Populate cut-player dropdown
  let cutSelect = document.getElementById("cutPlayer");
  cutSelect.innerHTML = players.map((player, index) => `
    <option value="${index}">${player}</option>
  `).join('');
  // Switch view from setup to game area
  document.getElementById("setup").classList.add("hidden");
  document.getElementById("gameArea").classList.remove("hidden");
  document.getElementById("roundInfo").textContent = "Round " + currentRound + " (Hand Size: " + handSize + ")";
  log("Game started with players: " + players.join(", "));
  updateProbabilitiesTable();
});

document.getElementById("submitDeclarations").addEventListener("click", async () => {
  // Read each player's declaration
  players.forEach((player, index) => {
    let val = parseInt(document.getElementById("decl_" + index).value);
    declarations[index] = isNaN(val) ? 0 : val;
  });
  log("Declarations: " + declarations.join(", "));
  // Calculate probabilities
  let stats = await Declaration(probabilitiesList, declarations, handSize, activeWires, posEvil, numBomb);
  probabilitiesList = stats.probs_list;
  prior = stats.prior;
  updateProbabilitiesTable(stats.wire, stats.bomb, stats.evil, stats.score);
  // Show the wire cutting controls after declarations are submitted
  document.getElementById("cutArea").classList.remove("hidden");
});

document.getElementById("submitCut").addEventListener("click", async () => {
  let cutIndex = parseInt(document.getElementById("cutPlayer").value);
  let result = parseInt(document.getElementById("wireResult").value);
  log(players[cutIndex] + "'s wire cut. Result: " + result);
  revealed[cutIndex] += 1;
  numWires -= 1;
  if (result === 1) {
    found[cutIndex] += 1;
    activeWires -= 1;
  }
  // Check victory condition: bomb detonated.
  if (result === 2) {
    log("The Bomb was detonated. Bad guys win!");
    alert("The Bomb was detonated. Bad guys win!");
    document.getElementById("cutArea").classList.add("hidden");
    return;
  }
  // Update probabilities
  let stats = await Cut(probabilitiesList, prior, declarations, revealed, found, handSize, activeWires, posEvil, numBomb);
  probabilitiesList = stats.probs_list;
  updateProbabilitiesTable(stats.wire, stats.bomb, stats.evil, stats.score);
  // Check victory condition: no active wires remain.
  if (activeWires <= 0) {
    log("All wires have been cut. Good guys win!");
    alert("All wires have been cut. Good guys win!");
    return;
  }
  // When a full round of cuts is done, start a new round.
  if (numWires <= players.length * (handSize - 1)) {
    // Check victory condition: end of last round.
    if (handSize === 2) {
      log("Out of time. Bad guys win!");
      alert("Out of time. Bad guys win!");
      return;
    }
    currentRound += 1;
    handSize -= 1;
    log("Starting round " + currentRound + " (Hand Size: " + handSize + ")");
    document.getElementById("roundInfo").textContent = "Round " + currentRound + " (Hand Size: " + handSize + ")";
    // Reset declarations for the new round
    players.forEach((player, index) => {
      document.getElementById("decl_" + index).value = "0";
    });
    // Hide cut controls until new declarations are submitted
    document.getElementById("cutArea").classList.add("hidden");
  }
});

// Sidebar toggle functionality
const toggleSidebar = document.getElementById('toggleSidebar');
const Sidebar = document.getElementById('Sidebar');

toggleSidebar.addEventListener('click', () => {
  Sidebar.classList.toggle('open');
  if (Sidebar.classList.contains('open')) {
    toggleSidebar.textContent = 'Hide Stats';
  } else {
    toggleSidebar.textContent = 'Show Stats';
  }
});

});

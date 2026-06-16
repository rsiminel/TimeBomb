from itertools import combinations
import matplotlib.pyplot as plt
from tabulate import tabulate
from random import randint
from copy import deepcopy
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import keras
import General as tb

def StrategyWrapper(actor):
  def Strategy(decls, probs_list, probs, revealed, found, hand_size, active_wires, cut, curr_cut, pos_bad, num_bom):
    num_players = decls.size
    prev_cutee = [0] * num_players
    prev_cutee[curr_cut] = 1
    p_wire = np.zeros(num_players)
    p_bomb = np.zeros(num_players)
    comb_probs = np.zeros(num_players)
    for i in range(len(pos_bad)):
      _, prob_bomb = tb.Separate(probs[i], pos_bad[i][0], num_bom)
      for bom_set in combinations(range(num_players), num_bom):
        for bom in bom_set:
          if hand_size - revealed[bom] != 0:
            p_bomb[bom] += pos_bad[i][1] * prob_bomb[bom_set] / (hand_size - revealed[bom])
      comb_probs += pos_bad[i][1] * tb.Flatten(tb.CombineProbs(probs_list[i]))
      total_probs = tb.CombineNonHomoProbs(tb.CombineProbs(probs_list[i][0:-1]), probs[i], pos_bad[i][0], num_bom)
      p_wire += pos_bad[i][1] * tb.P_wire(decls, total_probs, revealed, found, hand_size, active_wires + np.sum(found), pos_bad[i][0], num_bom)
    state = np.array(prev_cutee + list(comb_probs) + list(p_bomb) + list(p_wire) + [hand_size] + [cut] + [active_wires])
    # Update observations and actions
    obs_tensor = tf.expand_dims(state, axis=0)
    action = sample_action(actor, obs_tensor).numpy()[0]
    return action
  return Strategy

# Game Simulation
def run_episode(actor):
  """
  Runs one game (episode)
  Returns:
    observations: NumPy array of shape (episode_length, observation_dimensions)
    actions: NumPy array of shape (episode_length,) containing integer actions
    rewards: NumPy array of shape (episode_length,) containing rewards from each step
    is_win: 0 (loss) or 1 (win)
  """
  # Game Parameters
  num_players = 4
  pos_bad = [[1, 1.0]]
  num_bad = 1
  num_bom = 1
  initial_hand_size = 5
  # Reward Parameters
  wire_cut = 3
  bomb_cut = -3
  nada_cut = 0
  rule_cut = -10
  wire_game = 1
  bomb_game = -1
  nada_game = -1
  rule_game = -1
  # Initializations
  observations = []
  actions = []
  rewards = []
  hand_size = initial_hand_size
  num_wires = num_players * hand_size
  active_wires = num_players
  zeros = np.zeros(num_players, dtype=np.int8)
  # Distributing roles
  roles = zeros.copy()
  evil = 0
  while evil < num_bad:
    randy = randint(0, num_players - 1)
    if roles[randy] == 0:
      roles[randy] = 1
      evil += 1
  # Initialize probabilities
  probabilities_list = [[] for _ in range(len(pos_bad))]
  # Starting turns
  while hand_size > 1:
    # Distribute wires
    wires, bombs = tb.DistributeWires(num_players, hand_size, active_wires, num_bom)
    # Declare your wires
    declarations = wires.copy()
    for player in range(num_players):
      maxx = min(hand_size, active_wires)
      if roles[player] == 1:
        if bombs[player] == 1:
          declarations[player] = randint(wires[player], maxx)
        else:
          declarations[player] = randint(0, maxx)
      elif bombs[player] == 1:
        declarations[player] = randint(0, wires[player])
    # Calculate probabilities
    probabilities = [0 for _ in range(len(pos_bad))]
    prob_bad = [0 for _ in range(len(pos_bad))]
    for i in range(len(pos_bad)):
      probabilities[i] = tb.ProbDeclaration(declarations, hand_size, active_wires, pos_bad[i][0], num_bom)
      prob_bad[i], _ = tb.Separate(probabilities[i], pos_bad[i][0], num_bom)
      probabilities_list[i].append(deepcopy(prob_bad[i]))
    # Cut wires
    found = zeros.copy()
    revealed = zeros.copy()
    probs = deepcopy(probabilities)
    cutee = -1
    for cut in range(num_players):
      # Calculate state (mlp input)
      prev_cutee = [0] * num_players
      prev_cutee[cutee] = 1
      p_wire = np.zeros(num_players)
      p_bomb = np.zeros(num_players)
      comb_probs = np.zeros(num_players)
      for i in range(len(pos_bad)):
        _, prob_bomb = tb.Separate(probs[i], pos_bad[i][0], num_bom)
        for bom_set in combinations(range(num_players), num_bom):
          for bom in bom_set:
            if hand_size - revealed[bom] != 0:
              p_bomb[bom] += pos_bad[i][1] * prob_bomb[bom_set] / (hand_size - revealed[bom])
        comb_probs += pos_bad[i][1] * tb.Flatten(tb.CombineProbs(probabilities_list[i]))
        total_probs = tb.CombineNonHomoProbs(tb.CombineProbs(probabilities_list[i][0:-1]), probs[i], pos_bad[i][0], num_bom)
        p_wire += pos_bad[i][1] * tb.P_wire(declarations, total_probs, revealed, found, hand_size, active_wires + np.sum(found), pos_bad[i][0], num_bom)
      state = np.array(prev_cutee + list(comb_probs) + list(p_bomb) + list(p_wire) + [hand_size] + [cut] + [active_wires])
      # Update observations and actions
      observations.append(state.copy())
      obs_tensor = tf.expand_dims(state, axis=0)
      action = sample_action(actor, obs_tensor).numpy()[0]
      actions.append(action)
      new_cutee = action
      # Reveal a card
      if new_cutee == cutee or revealed[cutee] >= hand_size:
        rewards.append(rule_cut)
        return np.array(observations), np.array(actions), np.array(rewards) + rule_game, 0
      else: cutee = new_cutee
      randy = randint(1, hand_size - revealed[cutee])
      if bombs[cutee] == 1 and randy == hand_size - revealed[cutee]:
        rewards.append(bomb_cut)
        return np.array(observations), np.array(actions), np.array(rewards) + bomb_game, 0
      elif randy <= wires[cutee] - found[cutee]:
        found[cutee] += 1
        active_wires -= 1
        rewards.append(wire_cut)
      else:
        rewards.append(nada_cut)
      revealed[cutee] += 1
      num_wires -= 1
      # Update probabilities
      probs = [0 for _ in range(len(pos_bad))]
      prob_bad = [0 for _ in range(len(pos_bad))]
      for i in range(len(pos_bad)):
        probs[i] = tb.ProbCut(declarations, probabilities[i], revealed, found, hand_size, active_wires + np.sum(found), pos_bad[i][0], num_bom)
        prob_bad[i], _ = tb.Separate(probs[i], pos_bad[i][0], num_bom)
        probabilities_list[i][-1] = deepcopy(prob_bad[i])
      # Test for victory
      if active_wires <= 0:
        return np.array(observations), np.array(actions), np.array(rewards) + wire_game, 1
    # Next round
    hand_size -= 1
  return np.array(observations), np.array(actions), np.array(rewards) + nada_game, 0

# Define a simple MLP model for the actor
def mlp(x, sizes, activation=tf.nn.tanh, output_activation=None):
  for size in sizes[:-1]:
    x = tf.keras.layers.Dense(units=size, activation=activation)(x)
  return tf.keras.layers.Dense(units=sizes[-1], activation=output_activation)(x)

# Sampling an action from the actor's policy (categorical distribution)
def sample_action(actor, observation):  # observation is assumed to be a tensor of shape (batch, observation_dimensions)
  logits = actor(observation)
  action = tf.squeeze(tf.random.categorical(logits, num_samples=1), axis=1)
  return action

# Training Function Using REINFORCE
def train(actor, games_per_epoch, optimizer):
  all_observations = []
  all_actions = []
  all_returns = []
  wins = 0
  # Run several episodes to collect training data
  for _ in tqdm(range(games_per_epoch)):
    obs, actions, returns, win = run_episode(actor)
    all_observations.append(obs)
    all_actions.append(actions)
    all_returns.append(returns)
    wins += win
  # Flatten the episodes into a single batch
  observations = np.concatenate(all_observations, axis=0)
  actions = np.concatenate(all_actions, axis=0)
  returns = np.concatenate(all_returns, axis=0)
  # Convert to tensors
  observations = tf.convert_to_tensor(observations, dtype=tf.float32)
  actions = tf.convert_to_tensor(actions, dtype=tf.int32)
  returns = tf.convert_to_tensor(returns, dtype=tf.float32)
  # Compute the policy gradient loss
  with tf.GradientTape() as tape:
    # Forward pass: compute logits for all observations
    logits = actor(observations)  # shape: (N, num_players)
    log_probs = tf.nn.log_softmax(logits, axis=1)
    # Gather the log probabilities for the actions actually taken
    indices = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
    selected_log_probs = tf.gather_nd(log_probs, indices)
    # REINFORCE loss: weight the log probability by the return
    score = -tf.reduce_mean(selected_log_probs * returns)
  # Compute gradients and update the actor's parameters
  grads = tape.gradient(score, actor.trainable_variables)
  optimizer.apply_gradients(zip(grads, actor.trainable_variables))
  return score.numpy(), wins

def make_model(model_file, num_players):
  # Training Hyperparameters
  num_epochs = 100
  games_per_epoch = 10000
  observation_dimensions = 4 * num_players + 3
  hidden_sizes = [32, 8]
  optimizer = keras.optimizers.Adam(learning_rate=0.05)
  # Create actor
  observation_input = tf.keras.Input(shape=(observation_dimensions,), dtype="float32")
  logits = mlp(observation_input, hidden_sizes + [num_players])
  actor = tf.keras.Model(inputs=observation_input, outputs=logits)
  print(actor.summary())
  # Training Loop
  scores = np.zeros(num_epochs)
  for epoch in range(num_epochs):
    lr = optimizer.learning_rate.numpy()
    score, wins = train(actor, games_per_epoch, optimizer)
    table = [
      ["Epoch", "Wins", "Score", "LR"]] + [
      [epoch, wins, score, lr]
    ]
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid', floatfmt=(".0f", ".0f", ".3f", ".3f", ".3f")))
    scores[epoch] = score
    # if epoch == 25:
    #   games_per_epoch = 1000
    # if epoch == 50:
    #   games_per_epoch = 2500
    # if epoch == 75:
    #   games_per_epoch = 5000
    # if epoch == 100:
    #   games_per_epoch = 10000
    # if epoch == 125:
    #   games_per_epoch = 20000
  # Save Model
  actor.save(model_file)
  plt.plot(scores)
  plt.title('model score')
  plt.ylabel('score')
  plt.xlabel('epoch')
  plt.show()

def main(model_name):
  num_players = 4
  model_file = './models/' + model_name  + '.keras'
  make_model(model_file, num_players)
  actor = keras.models.load_model(model_file)
  # tb.PlayAuto(StrategyWrapper(actor), num_players)
  strategies = [StrategyWrapper(actor), tb.CutMaxScore, tb.CutRandom]
  win_rates, suspicion = tb.Test(strategies, 4, 5000)
  print(win_rates, suspicion)
  return

main("Test")

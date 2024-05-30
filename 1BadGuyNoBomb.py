# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 14:21:49 2022

@author: Remy
"""

# Imports
import numpy as np
from random import randint
import UsefulFunctions as uf


# 1 bad guy, no bomb
# Suppositions : good guys tell the truth, bad guys lie randomly
def PlayAuto(num_players=4, initial_hand_size=5):
    hand_size = initial_hand_size
    num_wires = num_players * hand_size
    active_wires = num_players
    # Distributing roles
    roles = np.zeros(num_players)
    roles[randint(0, num_players - 1)] = 1
    # Initialize probabilities
    probabilities_list = []
    # Starting turns
    while hand_size > 1:
        print("Round ", initial_hand_size - hand_size + 1)
        wires = uf.DistributeWires(num_players, active_wires, hand_size)
        print("w:", wires)
        # Declare your wires
        declarations = wires.copy()
        for i in range(num_players):
            if roles[i] == 1:
                randy = randint(- wires[i], hand_size - wires[i])
                declarations[i] += randy
        print("d:", declarations)
        # Calculate probabilities
        probabilities = ProbDeclaration(declarations, hand_size, active_wires)
        probabilities_list.append(probabilities.copy())
        print(" p:", probabilities)
        print("tp:", CombineProbs(probabilities_list))
        # Cut wires
        found = np.zeros(num_players)
        revealed = np.zeros(num_players)
        for i in range(num_players):
            print("Cut number", i + 1)
            cutee = randint(0, num_players - 1)  # cutee = max(prob, dclrtn)
            while revealed[cutee] >= hand_size:
                cutee = randint(0, num_players - 1)
            randy = randint(1, hand_size - revealed[cutee])
            if randy <= wires[cutee] - found[cutee]:
                found[cutee] += 1
                active_wires -= 1
            revealed[cutee] += 1
            num_wires -= 1
            print("r:", revealed)
            print("f:", found)
            # Update probabilities
            probs = ProbCut(declarations, probabilities, revealed, found,
                            hand_size, active_wires)
            probabilities_list[-1] = probs.copy()
            print("  p:", probs)
            print(" tp:", CombineProbs(probabilities_list))
            # Test for victory
            if active_wires <= 0:
                print("Good guys win!")
                return
        # Next round
        hand_size -= 1
        print("\n")
    print("Bad guys win!")
    return


def Play(players=["Alice", "Bob", "Clara", "Darryl"], initial_hand_size=5):
    num_players = len(players)
    hand_size = initial_hand_size
    num_wires = num_players * hand_size
    active_wires = num_players
    # Initialize probabilities
    probabilities_list = []
    # Starting turns
    while hand_size > 1:
        print("Round ", initial_hand_size - hand_size + 1)
        # Declare your wires
        declarations = np.zeros(num_players)
        for i in range(num_players):
            declarations[i] = int(input("How many wires does " +
                                        players[i] + " say they have? "))
        print("d:", declarations)
        # Calculate probabilities
        probabilities = ProbDeclaration(declarations, hand_size, active_wires)
        probabilities_list.append(probabilities.copy())
        print(" p:", probabilities)
        print("tp:", CombineProbs(probabilities_list))
        # Cut wires
        found = np.zeros(num_players)
        revealed = np.zeros(num_players)
        for i in range(num_players):
            print("Cut number", i + 1)
            cutee_str = input("Who's wire has been cut? ")
            while cutee_str not in players:
                cutee_str = input("You must have made a typo. Who? ")
            cutee = 0
            for j in range(num_players):
                if players[j] == cutee_str:
                    cutee = j
            revealed[cutee] += 1
            num_wires -= 1
            shown = int(input("Did you reveal an\n" +
                              " 1- inactive wire\n 2- active wire\n"))
            while shown not in [1, 2, 3]:
                shown = int(input("Sorry, I'm looking for a 1 or a 2 here."))
            if shown == 2:
                found[cutee] += 1
                active_wires -= 1
            print("r:", revealed)
            print("f:", found)
            # Update probabilities
            probs = ProbCut(declarations, probabilities, revealed, found,
                            hand_size, active_wires)
            probabilities_list[-1] = probs.copy()
            print(" p:", probs)
            print("tp:", CombineProbs(probabilities_list))
            # Test for victory
            if active_wires <= 0:
                print("Good guys win!")
                return
        # Next round
        hand_size -= 1
        print("\n")
    print("Bad guys win!")
    return


def ProbDeclaration(declarations, hand_size, active_wires):
    num_players = declarations.size
    probabilities = np.full(num_players, 1 / num_players)
    excess = - active_wires
    for declaration in declarations:
        excess += declaration
    if excess != 0:
        cas_p = 0
        for i in range(num_players):
            if excess > 0:
                probabilities[i] = uf.C(excess, declarations[i])
                cas_p += probabilities[i]
            else:
                probabilities[i] = uf.C(- excess, hand_size - declarations[i])
                cas_p += probabilities[i]
        probabilities /= cas_p
    return probabilities


def ProbCut(decls, probabilities, revealed, found, hand_size, active_wires):
    num_players = len(decls)
    probs = probabilities.copy()  # Don't modify the original array
    for i in range(num_players):
        if probs[i] == 1:  # The bad guy has already been found
            return probs
        # Probability that what happened would happen if i is a good guy
        penh = uf.Lklhd(hand_size, decls[i], revealed[i], found[i])
        # How many wires does i have if he is a bad guy?
        missing = active_wires + np.sum(found) - np.sum(decls) + decls[i]
        # Probability that what happened would happen if i were the bad guy
        peh = uf.Lklhd(hand_size, missing, revealed[i], found[i])
        # Update probability using Bayes formula
        new_prob = probs[i] * peh / (probs[i] * peh + (1 - probs[i]) * penh)
        # Update other probabilities to normalize the matrix
        for j in range(num_players):
            if j != i:
                probs[j] *= 1 - (new_prob - probs[i]) / (1 - probs[i])
        probs[i] = new_prob
    return probs


# This has empirically been show to give the same results as the other function
# But this one comes with a slightly more rigorous derrivation
def MathematicallyJustifiedProbCut(decls, probabilities, revealed, found,
                                   hand_size, active_wires):
    num_players = len(decls)
    probs = probabilities.copy()  # Don't modify the original array
    for i in range(num_players):
        if probs[i] == 1:  # Bad guy has already been found
            return probs
        # Probability that what happened would happen if i is a good guy
        penh = uf.Lklhd(hand_size, decls[i], revealed[i], found[i])
        # How many wires does i have if he is a bad guy?
        missing = active_wires + np.sum(found) - np.sum(decls) + decls[i]
        # Probability that what happened would happen if i were the bad guy
        peh = uf.Lklhd(hand_size, missing, revealed[i], found[i])
        # Update probability using Bayes formula
        new_prob = probs[i] * peh / (probs[i] * peh + (1 - probs[i]) * penh)
        for j in range(num_players):
            if probs[j] == 1:  # Bad guy has already been found
                return probs
            if j != i:
                # Prob of "i is bad" supposing that prob of "j is bad" is 0
                pispj0 = probs[i] * (1 + probs[j] / (1 - probs[j]))
                # Prob of what happend supposing j is good
                penhj = pispj0 * peh + (1 - pispj0) * penh
                # If j is bad then i is automatically good
                pehj = penh
                # Update probability using Bayes formula
                probs[j] = probs[j]*pehj / (probs[j]*pehj + (1-probs[j])*penhj)
        probs[i] = new_prob
    return probs


# Arithmetic average preserves 0s and 1s. Don't ask any other questions.
def CombineProbs(probabilities_list):
    num_tests = len(probabilities_list)
    num_players = len(probabilities_list[0])
    probabilities = np.full(num_players, 1.)
    for i in range(num_players):
        for j in range(num_tests):
            probabilities[i] *= probabilities_list[j][i]
    probabilities /= sum(probabilities)
    return probabilities

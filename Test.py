# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 12:42:27 2024

@author: Remy
"""

import OneBadGuyNoBomb as ibob
import TwoBadGuysNoBomb as tbob
import OneBadGuyOneBomb as ibib
import TwoBadGuysOneBomb as tbib


# 1 bad guy, no bomb
def Test1b0b(n=5, num_players=4, init_hand_size=5):
    win_rate = 0
    find_rate = 0
    suspicion = 0
    for i in range(n):
        (is_win, probs, roles) = ibob.PlayAuto(num_players, init_hand_size, 0)
        win_rate += is_win
        suspect = 0
        max_sus = 0
        culprit = 0
        for i in range(num_players):
            if probs[i] > max_sus:
                max_sus = probs[i]
                suspect = i
            if roles[i] == 1:
                culprit = i
        if suspect == culprit:
            find_rate += 1
        suspicion += probs[culprit]
    return (win_rate/n, find_rate/n, suspicion/n)


def Test(play_auto, num_players, num_games=5, init_hand_size=5):
    win_rate = 0
    suspicion = []
    for i in range(num_games):
        (is_win, probs, roles) = play_auto(num_players, init_hand_size, 0)
        win_rate += is_win
        culprits = []
        for i in range(num_players):
            if roles[i] == 1:
                culprits += [i]
        suspicion += [probs[culprits[j]] for j in range(len(culprits))]
    return (win_rate/num_games, sum(suspicion)/num_games/len(culprits))


num_tests = 1000
print(Test(ibob.PlayAuto, 4, num_tests))
print(Test(tbob.PlayAuto, 6, num_tests))
print(Test(ibib.PlayAuto, 4, num_tests))
print(Test(tbib.PlayAuto, 6, num_tests))

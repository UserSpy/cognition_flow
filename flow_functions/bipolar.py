import random

def main(inputString, tempHistory, previousBotReply, userReply, state):
    print("\nUsing bipolar") 
    if (random.randint(0, 1)):
        response = "in an extremely excited way"
    else:
        response = "in an extremely rude, angry, and mean way"
    return ('\nNow respond to this question or statement ' + response + ': ' + inputString), 1
def main(inputString, tempHistory, previousBotReply, userReply, state):
    print("\nUsing innerThoughtStarter")
    print("\n", state['new_generation'])
    return ('Conversation:\n' + state['name2'] + ': ' + previousBotReply + "\n" + state['name1'] + ': ' + userReply + '\n*Your Inner Thoughts about what was said:'), 1
def main(inputString, tempHistory, previousBotReply, userReply, state):

    return ('Conversation:\n' + state['name2'] + ': ' + previousBotReply + "\n" + state['name1'] + ': ' + userReply +'\nAs ' + state['name2'] + ', write a note to self that thinks about and analyzes ' + state['name1'] + '\'s response: '), 1
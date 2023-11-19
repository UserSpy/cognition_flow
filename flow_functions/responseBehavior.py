def main(inputString, tempHistory, previousBotReply, userReply, state):
    print("\nUsing responseBehavior") 
    print("\n", state['new_generation'])
    return (tempHistory + '\nNow respond to the original question or statement using your previous thoughts: ' + userReply), 1
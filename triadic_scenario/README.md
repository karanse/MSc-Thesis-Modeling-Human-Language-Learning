
- get the sibling agent(both listener and speaker) from pretrained saved model, (.pth files)
- get the child agent (both listener and speaker) from child_agent.py file
- get the data from data folder (dict_words_boxes.json  ha_vgg_indices.json  test_data.txt  train_data.txt  validation_data.txt) and also  ha_bbox_vggs (which is not in this repo)

- train the model (run the bash script --> run_tests.sh file):
   - train sibling and save the word_guess
   - mix the real language_input and word_guesses from sibling as child_language_input
   - train child with child_language_input
   
- save both sibling & child losses and accuracies
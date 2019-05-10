## Here are the main steps how triadic scenario works

- gets the sibling agent(both listener and speaker) from pretrained saved model, (.pth files)
- gets the child agent (both listener and speaker) from child_agent.py file
- gets the data from data folder (dict_words_boxes.json  ha_vgg_indices.json  test_data.txt  train_data.txt  validation_data.txt) and also  ha_bbox_vggs (which is not in this repo because of the size)

- trains the model (run the bash script --> run_tests.sh file):
   - trains sibling and save the word_guess
   - mixes the real language_input and word_guesses from sibling as child_language_input
   - trains child with child_language_input
   
- saves both sibling & child losses and accuracies

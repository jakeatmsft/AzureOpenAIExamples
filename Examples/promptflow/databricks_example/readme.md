Step 1:  Upload the "chat_csv_model" folder to your databricks workspace, along with the deploy_pf notebooks.

<img width="171" alt="image" src="https://github.com/jakeatmsft/AzureOpenAIExamples/assets/47987698/5b575c11-a8ac-4e8e-ac00-6447483f4f66">
</br>
Step 2: Open the pf_register_model.ipynb Run through the notebook to test the promptflow model and register it to your databricks model registry.
<b>Be sure to replace all connection string info with your AOAI config.</b>
<img width="278" alt="image" src="https://github.com/jakeatmsft/AzureOpenAIExamples/assets/47987698/64ac8725-e920-469a-a3b9-9dc02fe7d563">

After execution of all steps you should see the model registered:
</br>
<img width="242" alt="image" src="https://github.com/jakeatmsft/AzureOpenAIExamples/assets/47987698/0aed6511-3211-42fc-9d7d-59cd4b85aa0e">
</br>
Step 3: Open the pf_test_model.ipynb to load the model from the registry and ensure you can execute it successfully.

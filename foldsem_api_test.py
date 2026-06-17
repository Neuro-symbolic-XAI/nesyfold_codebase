import pandas as pd
import requests

def foldsem_api_test(train_filter_table_path, val_filter_table_path, test_filter_table_path, user, password):
    # Enter File Paths
    json_path = "foldsem_model.json" # Enter the model file path
    test_path_list = [train_filter_table_path, val_filter_table_path, test_filter_table_path] # Enter the test dataset paths as a comma seperated list

    with open(json_path, 'r') as f:
        model_json = f.read()

    test_df_json_list = []
    for test_path in test_path_list:
        test_df = pd.read_csv(test_path)
        test_df.columns = ['str_'+ str(x) for x in test_df.columns]
        try:
            test_data_frame_json = test_df.to_json(orient='split')
            test_df_json_list.append(test_data_frame_json)
        except:
            print("Error processing file at file path ",test_path)

    # Enter Your Username and Password
    username = user  # Enter your registered email id
    password = password  # Enter your password
    # If your username and password combination does not work, please try to register again with the same email id. Please contact us if that does not work.

    payload = {
        'username': username,
        'password': password,
        'test_data_frame_json_list': test_df_json_list,
        'json_model': model_json
    }

    # response = requests.post("http://foldse-loadbalancer-339280618.us-east-1.elb.amazonaws.com/auth/foldmodel_binary_json/", json=payload) # Uncomment if you want to run binary classification model.
    #"http://foldse-loadbalancer-339280618.us-east-1.elb.amazonaws.com/auth/foldmodel_multicategory_json/
    response = requests.post("http://ec2-52-0-60-249.compute-1.amazonaws.com/auth/foldmodel_multicategory_json/", json=payload) # Uncomment if you want to run multi-category classification model.

    try:
        response = response.json()
        acc_list = []
        out_list = []
        for response_obj in response:
            try:
                if(response_obj['error']==None):
                    # If there is no error the response_obj dictionary contains the following keys - accuracy, f1_score, precision, recall, test_results (List of test dataset results only if test dataset provided).
                    # print(response_obj['accuracy'])
                    acc_list.append(response_obj['accuracy'])

                    # print(response_obj['test_results'])
                    out_list.append(response_obj['test_results'])
                    # Similarly you can use response_obj['accuracy'], response_obj['f1_score'], response_obj['precision'], response_obj['recall'], response_obj['test_results'].
                else:
                    print('Error: ', response_obj['error'])
            except:
                print("There was an error processing your request.")
    except:
        print(response)
    return acc_list[0], acc_list[1], acc_list[2], out_list[0], out_list[1], out_list[2]
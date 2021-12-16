def keras_kfold_validation(X,y,K_count,input_dims):
    from sklearn.model_selection import KFold
    import tensorflow.keras
    from keras.models import Sequential
    from keras.layers import Dense
    from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_recall_curve,recall_score,precision_score
    kfold_validation=KFold(K_count)
    Fold=0
    train_dict={}
    test_dict={}
    classification_dict={}
    accuracy_dict={}
    count={}
    for i,j in kfold_validation.split(X):
        Fold+=1
        count[Fold]=[len(i),len(j)]
        print(f"Fold #{Fold}")
        X_train_kf=X.loc[i[0]:i[-1]]
        X_test_kf=X.loc[j[0]:j[-1]]
        y_train_kf=y.loc[i[0]:i[-1]]
        y_test_kf=y.loc[j[0]:j[-1]]
        train_dict[Fold]=[X_train_kf,y_train_kf]
        test_dict[Fold]=[X_test_kf,y_test_kf]
        model = Sequential()
        model.add(Dense(150, input_dim=input_dims, activation='relu'))
        model.add(Dense(150, activation='softmax'))
        model.add(Dense(150, activation='linear'))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        epochs_hist = model.fit(np.array(X_train), y_train, epochs=250, batch_size=25,  verbose=1)
        y_predict_kf = np.round(model.predict(np.array(X_test_kf)))
        classification_dict[Fold]=[y_test_kf,y_predict_kf]
        accuracy_dict[Fold]=accuracy_score(y_test_kf,y_predict_kf)
        return train_dict,test_dict,classification_dict,accuracy_dict,count

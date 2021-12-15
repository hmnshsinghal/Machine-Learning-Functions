def del_null_rows():
    for col in data_train:
        if data_train[col].isnull().sum()>0:
            data_train.dropna(subset=[col],axis=0,inplace=True)

def drop_null_cols(threshold,data_train):
    for col in data_train:
        if data_train[col].isnull().sum()/len(data_train) >=threshold:
            Drop_columns.append(col)
            data_train.drop(col ,axis='columns',inplace= True)
            data_test.drop(col ,axis='columns',inplace= True)

def del_null_rows(threshold,data_train):
    for col in data_train:
        if data_train[col].isnull().sum()/len(data_train)>threshold:
            data_train.dropna(subset=[col],axis=0,inplace=True)

def categorical_data_coversion(data_train):
    from sklearn.preprocessing import LabelEncoder
    label_encoder=LabelEncoder()
    for i in data_train.select_dtypes("object").columns:
        data_train[i]=data_train[i].astype('str')
        data_train[i]=label_encoder.fit_transform(data_train[i])



def correlated_features(threshold,correlated_matrix):
    correlated_dict={}
    for feature_1 in correlated_matrix:
        correlated_dict[feature_1]=[]
        for feature_2 in correlated_matrix:
            if correlated_matrix[feature_1][feature_2] < -(threshold)  :
                 if correlated_matrix[feature_1][feature_2]!=1:
                        correlated_dict[feature_1].append(feature_2)
                        correlated_dict[feature_1].append(correlated_matrix[feature_1][feature_2])
            if correlated_matrix[feature_1][feature_2] > threshold :
                if correlated_matrix[feature_1][feature_2]!=1:
                    correlated_dict[feature_1].append(feature_2)
                    correlated_dict[feature_1].append(correlated_matrix[feature_1][feature_2])
    null_key_value=[keys for keys,value in correlated_dict.items() if correlated_dict[keys]==[]]
    for keys in null_key_value:
        del correlated_dict[keys]
    return correlated_dict

def mean_encoding(df_train, df_test, categorical_vars):
    
    # temporary copy of the original dataframes
    df_train_temp = df_train.copy()
    df_test_temp = df_test.copy()
    
    # iterate over each variable
    for col in categorical_vars:
        
        # make a dictionary of categories, target-mean pairs
        target_mean_dict = df_train.groupby([col])['survived'].mean().to_dict()
        
        # replace the categories by the mean of the target
        df_train_temp[col] = df_train[col].map(target_mean_dict)
        df_test_temp[col] = df_test[col].map(target_mean_dict)
    
    # drop the target from the daatset
    df_train_temp.drop(['survived'], axis=1, inplace=True)
    df_test_temp.drop(['survived'], axis=1, inplace=True)
    
    # return  remapped datasets
    return df_train_temp, df_test_temp

def fill_mean_value(data_train):
    for column in data_train.columns:
        data_train.fillna(value=data_train[column].mean(),inplace=True)

def columnsvariableencoding(Data_Frame):
    column_variable_encoding={}
    columns_unique={}
    for i in Data_Frame.columns:
        columns_unique[i]=Data_Frame[i].unique()
        k=[]
        for j in range(len(columns_unique[i])):
            k.append(j)
        column_variable_encoding[i]={}
        var_list=columns_unique[i]
        column_encoding={}
        column_encoding = dict(zip(var_list, k))
        column_variable_encoding[i]=column_encoding
    return column_variable_encoding

def classication_model_testing(X,y):
    from sklearn.metrics import f1_score
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC, NuSVC, SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, SGDClassifier
    from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
    from yellowbrick.classifier import ClassificationReport,DiscriminationThreshold
    from sklearn.pipeline import Pipeline
    models = [SVC(gamma='auto'), NuSVC(gamma='auto'), LinearSVC(), SGDClassifier(max_iter=100, tol=1e-3), KNeighborsClassifier(),    LogisticRegression(solver='lbfgs'), LogisticRegressionCV(cv=3),BaggingClassifier(), ExtraTreesClassifier(n_estimators=300),    RandomForestClassifier(n_estimators=300)]
    try:
        for model in models:
            Model = Pipeline([('one_hot_encoder', OneHotEncoder()),('estimator', model)])
            visualiser=ClassificationReport(Model, classes=['unacc','acc','good','vgood'],cmap="YlGn",size=(600, 360))
            visualiser.fit(X, y)
            visualiser.score(X, y)
            visualiser.show()
    except Exception as e:
        print('Model is not Working ' + str(e))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

def preprocessed_dataset(x,y,test_size=0.2,random_status= 42):

    encode = LabelEncoder()
    y_encode = encode.fit_transform(y)

    x_train,x_test,y_train,y_test = train_test_split(x, y_encode, test_size=test_size,random_state=random_status,stratify=y_encode)


    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state= random_status
    )
    model.fit(x_train_scaled,y_train)

    return model, scaler, encode, x_train_scaled, x_test_scaled, y_train, y_test








def predict (model,feature):
    pred = model.predict(feature)
    pred = scaler.inverse_transform(pred)
    pred = np.expm1(pred)
    return pred

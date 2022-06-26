import numpy as np
import pandas as pd
from sklearn import preprocessing
from tensorflow import keras

import sys

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import (QMainWindow, QApplication,
                             QPushButton, QLabel, QLineEdit, QMessageBox)


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'Тест'
        self.left = 500
        self.top = 200
        self.width = 400
        self.height = 800
        self.initUI()
        self.initPredictionModel()

    def initPredictionModel(self):
        self.min_max_scaler = preprocessing.MinMaxScaler()
        houses_data = pd.read_csv('housepricedata.csv')
        self.load_dataset = houses_data.values
        self.model = keras.models.load_model("training4.h5")

    def initUI(self):
        xCoord1 = 20
        widgetHeight = 20
        separatorHeight = 20

        textBoxWidth = 350
        textBoxHeight = 20

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # LotArea
        self.labelLotArea = QLabel(self)
        self.labelLotArea.move(xCoord1, separatorHeight)
        self.labelLotArea.setText("Площадь участка (в квадратных футах)")
        self.labelLotArea.adjustSize()

        self.textboxLotArea = QLineEdit(self)
        self.textboxLotArea.move(xCoord1, separatorHeight + widgetHeight)
        self.textboxLotArea.resize(textBoxWidth, textBoxHeight)

        # OverallQual
        self.labelOverallQual = QLabel(self)
        self.labelOverallQual.move(xCoord1, separatorHeight * 3 + widgetHeight)
        self.labelOverallQual.setText("Общее качество (шкала от 1 до 10)")
        self.labelOverallQual.adjustSize()

        self.textboxOverallQual = QLineEdit(self)
        self.textboxOverallQual.move(xCoord1, separatorHeight * 3 + widgetHeight * 2)
        self.textboxOverallQual.resize(textBoxWidth, textBoxHeight)

        # OverallCond
        self.labelOverallCond = QLabel(self)
        self.labelOverallCond.move(xCoord1, separatorHeight * 5 + widgetHeight * 2)
        self.labelOverallCond.setText("Общее состояние (шкала от 1 до 10)")
        self.labelOverallCond.adjustSize()

        self.textboxOverallCond = QLineEdit(self)
        self.textboxOverallCond.move(xCoord1, separatorHeight * 5 + widgetHeight * 3)
        self.textboxOverallCond.resize(textBoxWidth, textBoxHeight)

        # TotalBsmtSF
        self.labelTotalBsmtSF = QLabel(self)
        self.labelTotalBsmtSF.move(xCoord1, separatorHeight * 7 + widgetHeight * 3)
        self.labelTotalBsmtSF.setText("Общая площадь подвала (в кв. футах)")
        self.labelTotalBsmtSF.adjustSize()

        self.textboxTotalBsmtSF = QLineEdit(self)
        self.textboxTotalBsmtSF.move(xCoord1, separatorHeight * 7 + widgetHeight * 4)
        self.textboxTotalBsmtSF.resize(textBoxWidth, textBoxHeight)

        # FullBath
        self.labelFullBath = QLabel(self)
        self.labelFullBath.move(xCoord1, separatorHeight * 9 + widgetHeight * 4)
        self.labelFullBath.setText("Количество полностью оборудованных ванных комнат")
        self.labelFullBath.adjustSize()

        self.textboxFullBath = QLineEdit(self)
        self.textboxFullBath.move(xCoord1, separatorHeight * 9 + widgetHeight * 5)
        self.textboxFullBath.resize(textBoxWidth, textBoxHeight)

        # HalfBath
        self.labelHalfBath = QLabel(self)
        self.labelHalfBath.move(xCoord1, separatorHeight * 11 + widgetHeight * 6)
        self.labelHalfBath.setText("Количество половинных ванных комнат")
        self.labelHalfBath.adjustSize()

        self.textboxHalfBath = QLineEdit(self)
        self.textboxHalfBath.move(xCoord1, separatorHeight * 11 + widgetHeight * 7)
        self.textboxHalfBath.resize(textBoxWidth, textBoxHeight)

        # BedroomAbvGr
        self.labelBedroomAbvGr = QLabel(self)
        self.labelBedroomAbvGr.move(xCoord1, separatorHeight * 13 + widgetHeight * 7)
        self.labelBedroomAbvGr.setText("Количество спален над землей")
        self.labelBedroomAbvGr.adjustSize()

        self.textboxBedroomAbvGr = QLineEdit(self)
        self.textboxBedroomAbvGr.move(xCoord1, separatorHeight * 13 + widgetHeight * 8)
        self.textboxBedroomAbvGr.resize(textBoxWidth, textBoxHeight)

        # TotRmsAbvGrd
        self.labelTotRmsAbvGrd = QLabel(self)
        self.labelTotRmsAbvGrd.move(xCoord1, separatorHeight * 15 + widgetHeight * 8)
        self.labelTotRmsAbvGrd.setText("Общее количество комнат над землей")
        self.labelTotRmsAbvGrd.adjustSize()

        self.textboxTotRmsAbvGrd = QLineEdit(self)
        self.textboxTotRmsAbvGrd.move(xCoord1, separatorHeight * 15 + widgetHeight * 9)
        self.textboxTotRmsAbvGrd.resize(textBoxWidth, textBoxHeight)

        # Fireplaces
        self.labelFireplaces = QLabel(self)
        self.labelFireplaces.move(xCoord1, separatorHeight * 17 + widgetHeight * 9)
        self.labelFireplaces.setText("Количество каминов")
        self.labelFireplaces.adjustSize()

        self.textboxFireplaces = QLineEdit(self)
        self.textboxFireplaces.move(xCoord1, separatorHeight * 17 + widgetHeight * 10)
        self.textboxFireplaces.resize(textBoxWidth, textBoxHeight)

        # GarageArea
        self.labelGarageArea = QLabel(self)
        self.labelGarageArea.move(xCoord1, separatorHeight * 19 + widgetHeight * 10)
        self.labelGarageArea.setText("Площадь гаража (в кв. футах)")
        self.labelGarageArea.adjustSize()

        self.textboxGarageArea = QLineEdit(self)
        self.textboxGarageArea.move(xCoord1, separatorHeight * 19 + widgetHeight * 11)
        self.textboxGarageArea.resize(textBoxWidth, textBoxHeight)

        self.buttonPrediction = QPushButton('Сделать предсказание', self)
        self.buttonPrediction.move(xCoord1 - 10, separatorHeight * 21 + widgetHeight * 11)
        self.buttonPrediction.adjustSize()
        self.buttonPrediction.clicked.connect(self.on_click_button_prediction)

        self.buttonClearFields = QPushButton('Очистить поля', self)
        self.buttonClearFields.move(xCoord1 - 10, separatorHeight * 23 + widgetHeight * 11)
        self.buttonClearFields.adjustSize()
        self.buttonClearFields.clicked.connect(self.on_click_button_clear_fields)

        self.show()

    @pyqtSlot()
    def on_click_button_prediction(self):
        lotArea = int(self.textboxLotArea.text())
        overallQual = int(self.textboxOverallQual.text())
        overallCond = int(self.textboxOverallCond.text())
        totalBsmtSF = int(self.textboxTotalBsmtSF.text())
        fullBath = int(self.textboxFullBath.text())
        halfBath = int(self.textboxHalfBath.text())
        bedroomAbvGr = int(self.textboxBedroomAbvGr.text())
        totRmsAbvGrd = int(self.textboxTotRmsAbvGrd.text())
        fireplaces = int(self.textboxFireplaces.text())
        garageArea = int(self.textboxGarageArea.text())

        predict_dataset = np.concatenate((self.load_dataset, [[
            lotArea,
            overallQual,
            overallCond,
            totalBsmtSF,
            fullBath,
            halfBath,
            bedroomAbvGr,
            totRmsAbvGrd,
            fireplaces,
            garageArea,
            0
        ]]), axis=0)

        predict_X = predict_dataset[:, 0:10]
        predict_X_scale = self.min_max_scaler.fit_transform(predict_X)

        predict_vals = [[]]
        for curVal in predict_X_scale[-1]:
            predict_vals[0].append(curVal)

        predict_result = self.model.predict(predict_vals)

        if predict_result >= 0.5:
            normalize_predict_result = round((predict_result[0][0] - 0.5) * 2 * 100, 2)
            string_predict = "Цена выше рынка"
        else:
            normalize_predict_result = round((1.0 - predict_result[0][0] * 2) * 100, 2)
            string_predict = "Цена ниже рынка"

        output = string_predict + "  вероятность в % " + str(normalize_predict_result)

        QMessageBox.question(self, "Ответ", output, QMessageBox.Ok, QMessageBox.Ok)

    @pyqtSlot()
    def on_click_button_clear_fields(self):
        self.textboxLotArea.setText("")
        self.textboxLotArea.setText("")
        self.textboxOverallQual.setText("")
        self.textboxOverallCond.setText("")
        self.textboxTotalBsmtSF.setText("")
        self.textboxFullBath.setText("")
        self.textboxHalfBath.setText("")
        self.textboxBedroomAbvGr.setText("")
        self.textboxTotRmsAbvGrd.setText("")
        self.textboxFireplaces.setText("")
        self.textboxGarageArea.setText("")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())

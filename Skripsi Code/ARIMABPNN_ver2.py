import os
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
plt.rcParams['figure.figsize'] = 10, 4
import warnings
warnings.filterwarnings("ignore")

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_random_state
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from keras import optimizers
from keras.utils.vis_utils import plot_model

class ARIMAModel:
    def __init__(self, df, MaxDiff = 5, period = 12, fold = 4, p_value = 1, d_value = 1, q_value = 1):
        self.df = df
        self.MaxDiff = MaxDiff
        self.period = period
        self.fold = fold
        self.p_value = p_value
        self.d_value = d_value
        self.q_value = q_value

        self.OrderList = []
        self.DfResultList = []

        self.TrainForecast = []
        self.TestForecast = []

        self.TrainForecastList = []
        self.TestForecastList = []

    def CrossValidation(self, title = None, figsize = (15, 8)):
        # Make Benchmark
        TrainList = []
        TestList = []
        CVTable = {'Latih' : [],
                   'Uji'   : []}

        tscv = TimeSeriesSplit(gap = 0,
                               n_splits = self.fold,
                               max_train_size = None)
  
        fig, ax = plt.subplots(nrows = self.fold,
                               ncols = 1,
                               figsize = figsize)
        
        if title is None:
            pass
        elif title is not None:
            if isinstance(title, str):
                plt.suptitle(title)
            else:
                print("Input Judul Harus String")

        plt.suptitle(f'Validasi Silang Data Runtun Waktu\n(n_fold = {self.fold})', fontsize = 18)

        for i, (TrainIndex, TestIndex) in enumerate(tscv.split(self.df.iloc[:,0])):
            dfTrainIndex = self.df.iloc[TrainIndex]
            dfTestIndex = self.df.iloc[TestIndex]
            
            TrainList.append(dfTrainIndex)
            TestList.append(dfTestIndex)

            CVTable['Latih'].append(len(dfTrainIndex))
            CVTable['Uji'].append(len(dfTestIndex))

            ax[i].plot(dfTrainIndex, 
                       label = f"Data Latih {i+1}")
            ax[i].plot(dfTestIndex, 
                       label = f"Data Uji {i+1}")
            ax[i].set_title(f"Data Latih dan Data Uji Fold ke-{i+1}", 
                            fontsize = 14)
            ax[i].legend(loc = 'lower left')
            ax[i].grid(alpha = 0.5)
        
        plt.tight_layout()
        plt.show()

        CVTable = pd.DataFrame(CVTable)
        CVTable = CVTable.set_index(pd.Index([f'iterasi ke-{i+1}' for i in range(self.fold)]))

        return TrainList, TestList, CVTable

    def CheckStationarity(self, df = None, show_plot = True, graph_label = 'Data Asli'):

        if df is None:
            df = self.df
        
        rolling_mean = df.rolling(window = self.period).mean()

        rolling_std = df.rolling(window = self.period).std()

        if show_plot == True:
            plt.plot(df, 
                     color = 'blue', 
                     label = graph_label)
            plt.plot(rolling_mean, 
                     color = 'red', 
                     label = 'Rata-rata Bergerak')
            plt.plot(rolling_std, 
                     color = 'black', 
                     label = 'Standar Deviasi')
            plt.legend()
            plt.show()
        
        print('Results of Dickey-Fuller Test:')
        df_test = adfuller(df, autolag = 'AIC')
        df_output = pd.Series(df_test[0:4], 
                              index = ['Test Statistic', 
                                       'p-value', 
                                       '#Lags Used', 
                                       'Number of Observations Used'])
        for key, value in df_test[4].items():
            df_output['Critical Value (%s)' % key] = '{:.3f}'.format(value)
        print(df_output)

        if df_test[1] > 0.05:
            print('\nData tidak stasioner\n')
        else:
            print('\nData sudah stasioner\n')
        
    def CheckStationarityIter(self, dfList, show_plot = True, graph_label = 'Data Asli'):
        
        for Train in dfList:
            self.CheckStationarity(df = Train,
                                   show_plot = show_plot,
                                   graph_label = graph_label)

    def MakeStationary(self, df = None, show = True):

        if df is None:
            df = self.df
        
        for d in range(self.MaxDiff):
            if d == 0:
                pass
            else:
                DiffData = DiffData.diff()
                DiffData = DiffData[1:]
            
            rolmean = DiffData.rolling(window = self.period).mean()
            rolstd = DiffData.rolling(window = self.period).std()
            adf_result = adfuller(DiffData)

            # Cek apakah p-value memenuhi hipotesis nol
            if adf_result[1] < 0.05:
                print(f"Data sudah stasioner pada differencing ke {d} kali\n")
                print(f"p-value = {adf_result[1]}\n")
                break
        
        if show == False:
            pass
        elif show == True:
            self.CheckStationarity(DiffData,
                                   self.period,
                                   graph_label = 'Data Hasil Differensiasi')
        
        return DiffData

    def MakeStationaryIter(self, dfList, show = True):
        DiffDataList = []

        for i, data in enumerate(dfList):
            DiffData = self.MakeStationary(df = data)

            DiffDataList.append(DiffData)
        
        return DiffDataList

    def GridSearch(self, df, p_value = 0, d_value = 0, q_value = 0, print_details = False):
        
        if p_value == 0:
            p_value =self.p_value
        if d_value == 0:
            d_value =self.d_value
        if q_value == 0:
            q_value =self.q_value
        
        if df is None:
            df = self.df

        BestAic, BestBic, BestMle = np.inf, np.inf, np.inf
        BestARIMAAic, BestARIMABic, BestARIMAMle = np.inf, np.inf, np.inf
        CurrAic, CurrBic, CurrMle = np.inf, np.inf, np.inf
        BestOrder, BestOrderAic, BestOrderBic, BestOrderMle = None, None, None, None
        df_results = pd.DataFrame(columns = ['ARIMA', 'AIC', 'BIC', 'MLE'])
        
        for p in range(p_value + 1):
            for d in range(d_value + 1):
                for q in range(q_value + 1):
                    try:
                        model = ARIMA(df, order = (p, d, q))
                        model_fit = model.fit()

                        CurrAic = model_fit.aic
                        CurrBic = model_fit.bic
                        CurrMle = model_fit.mle_retvals['fopt']

                        df_results = df_results.append({'ARIMA' : f'({p},{d},{q})',
                                                        'AIC'   : f'{CurrAic}',
                                                        'BIC'   : f'{CurrBic}',
                                                        'MLE'   : f'{CurrMle}'},
                                                       ignore_index = True)
                        
                        if print_details is True:
                            print(f'Ordo ARIMA: {p,d,q}, nilai AIC: {CurrAic}, nilai BIC: {CurrBic}, nilai MLE: {CurrMle}')

                        if (CurrAic < BestAic) & (CurrBic < BestBic) & (CurrMle < BestMle):
                            BestAic = CurrAic
                            BestBic = CurrBic
                            BestMle = CurrMle
                            BestOrder = (p, d, q)
                    
                        if CurrAic < BestARIMAAic:
                            BestARIMAAic = CurrAic
                            BestOrderAic = (p, d, q)

                        if CurrBic < BestARIMABic:
                            BestARIMABic = CurrBic
                            BestOrderBic = (p, d, q)

                        if CurrMle < BestARIMAMle:
                            BestARIMAMle = CurrMle
                            BestOrderMle = (p, d, q)

                    except:
                        continue
        
        print(f'\nOrdo ARIMA terbaik berdasarkan data adalah {BestOrder} dengan nilai AIC {BestAic} | nilai BIC {BestBic} | nilai MLE {BestMle}')
        print(f'\nOrdo ARIMA terbaik berdasarkan data adalah {BestOrderAic} dengan nilai AIC {BestARIMAAic}')
        print(f'\nOrdo ARIMA terbaik berdasarkan data adalah {BestOrderBic} dengan nilai BIC {BestARIMABic}')
        print(f'\nOrdo ARIMA terbaik berdasarkan data adalah {BestOrderMle} dengan nilai MLE {BestARIMAMle}')

        return BestOrder, BestOrderAic, BestOrderBic, BestOrderMle, df_results
    
    def GridSearchIter(self, DataList, p_value = 1, d_value = 1, q_value = 1, details = False):
        # Make Benchmark
        OrderList, DfResultList = [], []

        if p_value == 0:
            p_value = self.p_value
        if d_value == 0:
            d_value == self.d_value
        if q_value == 0:
            q_value = self.q_value

        for i, df in enumerate(DataList):
            BestOrder, BestOrderAic, BestOrderBic, BestOrderMle, df_results = self.GridSearch(df.values,
                                                                                              p_value,
                                                                                              d_value,
                                                                                              q_value,
                                                                                              print_details = details)
            
            ARIMAOrder = (1, 1, 1)
            OrderListTemp = [BestOrder, BestOrderAic, BestOrderBic, BestOrderMle, ARIMAOrder]

            for order in OrderListTemp:
                if order not in OrderList:
                    OrderList.append(order)
            
            for i in range(len(OrderList)):
                for j in range(len(OrderList) - i - 1):
                    if OrderList[j] > OrderList[j + 1]:
                        OrderList[j], OrderList[j + 1] = OrderList[j], OrderList[j + 1]
            
            DfResultList.append(df_results)

            return OrderList, DfResultList

    def ARIMAForecastIter(self, OrderList, TrainList, TestList):
        # Benchmark
        TrainForecList, TestForecList = [], []

        for order in OrderList:
            TrainForecArr, TestForecArr = [], []

            for (i, (train, test)) in enumerate(zip(TrainList, TestList)):
                if i == 0:
                    model = ARIMA(train, order = order)
                    model_fit = model.fit()

                    TrainForec = pd.DataFrame(model_fit.predict())
                    TrainForec.columns = ['Peramalan']

                    TestForec = pd.DataFrame(model_fit.predict(start  = len(train)+1,
                                                               end    = len(train)+len(test)))
                    TestForec.index = test.index
                    TestForec.columns = ['Peramalan']

                elif i > 0:
                    model = ARIMA(train,
                                  order = order,
                                  enforce_stationarity = False,
                                  enforce_invertibility = False)
                
                    model_fit = model.filter(model_fit.params)

                    TrainForec = pd.DataFrame(model_fit.predict())
                    TrainForec.columns = ['Peramalan']

                    TestForec = pd.DataFrame(model_fit.predict(start  = len(train)+1,
                                                               end    = len(train)+len(test)))
                    TestForec.index = test.index
                    TestForec.columns = ['Peramalan']
                
                name = ''
                for j in order:
                    name += str(j)
                
                TrainForecArr.append(TrainForec)
                TestForecArr.append(TestForec)

                del mpdel
            
            TrainForecList.append(TrainForecArr)
            TestForecList.append(TestForecArr)

        return TrainForecList, TestForecList
            
    def GetResiduData(self, DfTrain = None, DfTest = None, DfTrainForec = None, DfTestForec = None):

        if (DfTrain is not None) and (DfTrainForec is not None):
            DfResidTrain = pd.DataFrame(DfTrain.iloc[:,0].sub(DfTrainForec.iloc[:,0],
                                                              fill_value = 0))
            
        if (DfTest is not None) and (DfTestForec is not None):
            DfResidTest = pd.DataFrame(DfTest.iloc[:,0].sub(DfTestForec.iloc[:,0],
                                                            fill_value = 0))
            
        return DfResidTrain, DfResidTest

    def GetResiduDatiter(self, DfTrain, DfTest, DfTrainForec, DfTestForec):
        
        DfResidTrainList, DfResidTestList = [], []

        ForecLists = zip(DfTrainForec,
                         DfTestForec)
        
        for (TrainArr, TestArr) in ForecLists:

            ActualArr = zip(DfTrain,
                            DfTest)
            
            ForecArr = zip(TrainArr,
                           TestArr)
            
            TrainResiduArr, TestResiduArr = [], []

            for i, (Actual, Forecast) in enumerate(zip(ActualArr, ForecArr)):
                
                Train, Test = Actual
                TrainForec, TestForec  = Forecast

                DfResidTrain, DfResidTest = self.GetResiduData(DfTrain = Train,
                                                                         DfTest = Test,
                                                                         DfTrainForec = TrainForec,
                                                                         DfTestForec = TestForec)
                
                TrainResiduArr.append(DfResidTrain)
                TestResiduArr.append(DfResidTest)
            
            DfResidTrainList.append(TrainResiduArr)
            DfResidTestList.append(TestResiduArr)
        
        return DfResidTrainList, DfResidTestList

class BPNNModel:
    def __init__(self):
        pass

    def SetSeed(self):
        pass

    def GetNormalizedDataIter(self):
        pass

    def GetIndex(self):
        pass

    def SlidingWindow(self):
        pass

    def SlidingWindowIter(self):
        pass

    def MakeModel(self):
        pass

    def ResidForecastIter(self):
        pass

class PlotAndEvaluation:
    def __init__(self):
        pass

    def ACFPACFSubplot(self, df, figsize = (20, 20)):

        fig, ax = plt.subplots(nrows = len(df),
                               ncols = 2,
                               figsize = figsize)
        fig.suptitle('Plot ACF dan PACF Data Hasil Differensiasi', 
                     fontsize = 20)
        
        x, y = 0, 0
        for i, DiffData in enumerate(df):
            plot_acf(DiffData,
                     ax = ax[x, y])
            plot_pacf(DiffData,
                      ax = ax[x, y+1])
            
            ax[x, y].set_title(DiffData, 
                               fontsize = 14)
            ax[x, y+1].set_title(DiffData, 
                                 fontsize = 14)
            x+=1
        
        plt.show()

    def GetPlot_oneline():
        pass

    def GetPlot_twoline():
        pass

    def GetEvaluation():
        pass

    def GetEvaluationIter():
        pass

    def GetEvaluationBarplot():
        pass

    def GetMeanEvaluation():
        pass

    def GetMeanEvaluationIter():
        pass

    def GetMeanEvaluationBarPlot():
        pass
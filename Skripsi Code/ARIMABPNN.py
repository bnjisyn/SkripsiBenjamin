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

        self.TrainList = []
        self.TestList = []

        self.CVTable = {'Latih'  : [],
                        'Uji'    : []}
        
        self.DiffData = []
        self.DiffDataList = []
        self.OrderList = []
        self.DfResultList = []

        self.TrainForecast = []
        self.TestForecast = []

        self.TrainForecastList = []
        self.TestForecastList = []
        
        self.EvalDfList = []
        self.EvalDfListMean = []

        self.DfResidTrainList = []
        self.DfResidTestList = []

    def CrossValidation(self, title = None):

        tscv = TimeSeriesSplit(gap = 0,
                               n_splits = self.fold,
                               max_train_size = None)
  
        fig, ax = plt.subplots(nrows = self.fold,
                               ncols = 1,
                               figsize = (15, 8))
        
        if title is None:
            pass
        elif title is not None:
            if isinstance(title, str):
                plt.suptitle(title)
            else:
                print("Input Judul Harus String")
        
        plt.suptitle(f'Validasi Silang Data Runtun Waktu\n(n_fold = {self.fold})', 
                     fontsize = 18)
        
        for i, (TrainIndex, TestIndex) in enumerate(tscv.split(self.df.iloc[:,0])):

            DfTrainName = f'DfTrain{i+1}'
            DfTestName  = f'DfTest{i+1}'

            locals()[DfTrainName] = self.df.iloc[TrainIndex]
            locals()[DfTestName] = self.df.iloc[TestIndex]

            self.TrainList.append(locals()[DfTrainName])
            self.TestList.append(locals()[DfTestName])

            self.CVTable['Latih'].append(len(locals()[DfTrainName]))
            self.CVTable['Uji'].append(len(locals()[DfTestName]))

            ax[i].plot(self.df.iloc[TrainIndex], label = f"data latih {i+1}")
            ax[i].plot(self.df.iloc[TestIndex], label = f"data uji {i+1}")
            ax[i].set_title(f"Data Latih dan Data Uji Fold ke-{i+1}", fontsize = 14)
            ax[i].legend(loc = 'lower left')
            ax[i].grid(alpha = 0.5)
        
        plt.tight_layout()
        plt.show()

        self.CVTable = pd.DataFrame(self.CVTable)
        self.CVTable = self.CVTable.set_index(pd.Index([f'iterasi ke-{i+1}' for i in range(self.fold)]))
        
        return self.TrainList, self.TestList, self.CVTable
    
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
    
    def CheckStationarityIter(self, dfList = None, show_plot = True, graph_label = 'Data Asli'):
        
        if dfList is None:
            dfList = self.TrainList

        for Train in dfList:
            self.CheckStationarity(df = Train,
                                   show_plot = show_plot,
                                   graph_label = graph_label)
    
    def MakeStationary(self, df = None, show = False):
        
        if df is None:
            df = self.df
        DiffData = df.copy()

        for d in range(self.MaxDiff):
            if d == 0:
                pass
            else:
                DiffData = DiffData.diff()
                DiffData = DiffData[1:]
            
            rolmean = DiffData.rolling(window = self.period).mean()
            rolstd = DiffData.rolling(window = self.period).std()
            adf_result = adfuller(DiffData)

            if adf_result[1] < 0.05:
                print(f"Data sudah stasioner pada differencing ke {d} kali\n")
                print(f"p-value = {adf_result[1]}\n")
                break
        
        if show == False:
            pass
        elif show == True:
            self.CheckStationarity(DiffData, self.period, graph_label = 'Data Hasil Differensiasi')

        self.DiffData = DiffData

        return self.DiffData
    
    def MakeStationaryIter(self, dfList = None, show = False):
        if dfList is None:
            dfList = self.TrainList

        for i, data in enumerate(dfList):
            DiffData = self.MakeStationary(df = data)
            
            DiffDataName = f'DiffDataCV{i+1}'
            locals()[DiffDataName] = DiffData
            self.DiffDataList.append(locals()[DiffDataName])
        
        return self.DiffDataList
    
    def ACFPACFSubplot(self):
        
        fig, ax = plt.subplots(nrows = len(self.DiffDataList),
                               ncols = 2,
                               figsize = (20,20))
        
        fig.suptitle('Plot ACF dan PACF Data Hasil Differensiasi', fontsize = 20)

        x, y = 0, 0
        for i, DiffData in enumerate(self.DiffDataList):
            plot_acf(DiffData, ax = ax[x, y])
            plot_pacf(DiffData, ax = ax[x, y+1])
            ax[x, y].set_title(f'Plot ACF (Autocorrelation Function Data Hasil Differensiasi iterasi ke-{i+1}',
                               fontsize = 14)
            ax[x, y+1].set_title(f'Plot PACF (Partial Autocorrelation Function Data Hasil Differensiasi iterasi ke-{i+1}',
                                 fontsize = 14)
            x+=1
        
        plt.show()
        
    def GridSearch(self, df = None, p_value = 0, d_value = 0, q_value = 0, print_details = True):
        
        if p_value == 0:
            p_value = self.p_value
        if d_value == 0:
            d_value == self.d_value
        if q_value == 0:
            q_value = self.q_value

        if df is None:
            df = self.df
        
        BestAic, BestBic, BestMle = np.inf, np.inf, np.inf
        BestARIMAAic, BestARIMABic, BestARIMAMle = np.inf, np.inf, np.inf

        CurrAic, CurrBic, CurrMle = np.inf, np.inf, np.inf

        BestOrder, BestOrderAic, BestOrderBic, BestOrderMle = None, None, None, None

        df_results = pd.DataFrame(columns = ['ARIMA', 'AIC', 'BIC', 'MLE'])

        for p in range(p_value+1):
            for d in range(d_value+1):
                for q in range(q_value+1):
                    try:
                        model = ARIMA(df, order=(p, d, q))
                        model_fit = model.fit()

                        CurrAic = model_fit.aic
                        CurrBic = model_fit.bic
                        CurrMle = model_fit.mle_retvals['fopt']

                        df_results = df_results.append({'ARIMA': f'({p},{d},{q})', 
                                                        'AIC': CurrAic, 
                                                        'BIC': CurrBic, 
                                                        'MLE': CurrMle}, 
                                                        ignore_index = True)

                        if print_details == True:
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

    def GridSearchIter(self, DataList = None, p_value = 1, d_value = 1, q_value = 1, details = False):

        if p_value == 0:
            p_value = self.p_value
        if d_value == 0:
            d_value == self.d_value
        if q_value == 0:
            q_value = self.q_value

        if DataList is None:
            if len(self.DiffDataList) == 0:
                DataList = self.TrainList
            elif len(self.DiffDataList) != 0:
                DataList = self.DiffDataList

        for i, df in enumerate(DataList):
            BestOrder, BestOrderAic, BestOrderBic, BestOrderMle, df_results = self.GridSearch(df.values,
                                                                                              p_value,
                                                                                              d_value,
                                                                                              q_value,  
                                                                                              print_details = details)
            
            ARIMAOrder = (1, 1, 1)
            OrderListTemp = [BestOrder, BestOrderAic, BestOrderBic, BestOrderMle, ARIMAOrder]

            for order in OrderListTemp:
                if order not in self.OrderList:
                    self.OrderList.append(order)
            
            for i in range(len(self.OrderList)):
                for j in range(len(self.OrderList)-i-1):
                    if self.OrderList[j] > self.OrderList[j+1]:
                        self.OrderList[j], self.OrderList[j+1] = self.OrderList[j+1], self.OrderList[j]

            DfResultName = f'DfResult{i+1}'
            locals()[DfResultName] = df_results
            self.DfResultList.append(locals()[DfResultName])

        return self.OrderList, self.DfResultList
    
    def ARIMAForecastIter(self):
        
        for order in self.OrderList:
            TrainForecArr, TestForecArr = [], []
        
            for (i, (train, test)) in enumerate(zip(self.TrainList, self.TestList)):
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
            
                TrainForecName = f'TrainForec{name}_iter{i+1}'
                TestForecName = f'TestForec{name}_iter{i+1}'

                locals()[TrainForecName] = TrainForec
                locals()[TestForecName] = TestForec

                TrainForecArr.append(locals()[TrainForecName])
                TestForecArr.append(locals()[TestForecName])
        
            self.TrainForecastList.append(TrainForecArr)
            self.TestForecastList.append(TestForecArr)

            del model


        return self.TrainForecastList, self.TestForecastList
    
    def ShowSubplotIter(self, Err = 2, Comb = False, LabelActual = 'Data Asli', LabelForec = 'Data Peramalan', figsize = (18,10),
                        DfTrainList = None, DfTestList = None,  DfTrainForecList = None, DfTestForecList = None):
        
        if (DfTrainList is None) and (DfTestList is None) and (DfTrainForecList is None) and (DfTestForecList is None):
        # if not any((DfDataList, DfTrainList, DfTestList, DfDataForecList, DfTrainForecList, DfTestForecList)):
            DfTrainList = self.TrainList
            DfTestList = self.TestList
            DfTrainForecList = self.TrainForecastList
            DfTestForecList = self.TestForecastList
        
        rows = int(self.fold)

        ForecastLists = zip(DfTrainForecList,
                            DfTestForecList)
        
        for order, ForecastArr in zip(self.OrderList, ForecastLists):
            TrainForecastArr, TestForecastArr = ForecastArr

            fig, ax = plt.subplots(nrows = rows,
                                   ncols = 2,
                                   figsize = figsize)
            
            if Comb == True:
                fig.suptitle(f'Peramalan Data dengan Kombinasi ARIMA - Backpropagation {order}', fontsize = 20)
            else:
                fig.suptitle(f'Peramalan Data dengan Model ARIMA {order}', fontsize = 20)
            
            for i in range(self.fold):
                ax[i, 0].plot(DfTrainList[i].iloc[Err:], 
                              label = LabelActual, 
                              color = 'blue')
                ax[i, 0].plot(TrainForecastArr[i].iloc[Err:],
                              label = LabelForec,
                            color = 'red')
                ax[i, 0].set_title(f'Peramalan Data Latih Fold ke-{i+1}',
                                   fontsize = 18)

                ax[i, 1].plot(DfTestList[i],
                              label = LabelActual,
                              color = 'blue')
                ax[i, 1].plot(TestForecastArr[i],
                              label = LabelForec,
                              color = 'red')
                ax[i, 1].set_title(f'Peramalan Data Uji Fold ke-{i+1}',
                                   fontsize = 18)
            
            plt.subplots_adjust(top = 0.88,
                                hspace = 0.5,
                                wspace = 0.1)
            plt.show()
            print('\n\n')
    
    def GetEvaluation(self, DfTrain = None, DfTest = None, DfTrainForec = None, DfTestForec = None):
        EvalDf = {}

        if (DfTrain is not None) and (DfTrainForec is not None):
            MSETrain = mean_squared_error(DfTrain, DfTrainForec, squared = False)
            MAETrain = mean_absolute_error(DfTrain, DfTrainForec)
            MAPETrain = mean_absolute_percentage_error(DfTrain, DfTrainForec)

            MSETrain = round(MSETrain, 3)
            MAETrain = round(MAETrain, 3)
            MAPETrain = round(MAPETrain, 3)

            EvalDf.update({'Latih' : [MSETrain, MAETrain, MAPETrain]})
        
        if (DfTest is not None) and (DfTestForec is not None):
            MSETest = mean_squared_error(DfTest, DfTestForec, squared = False)
            MAETest = mean_absolute_error(DfTest, DfTestForec)
            MAPETest = mean_absolute_percentage_error(DfTest, DfTestForec)

            MSETest = round(MSETest, 3)
            MAETest = round(MAETest, 3)
            MAPETest = round(MAPETest, 3)

            EvalDf.update({'Uji' : [MSETest, MAETest, MAPETest]})
        
        if EvalDf:
            EvalDf = pd.DataFrame(EvalDf, index = ['MSE', 'MAE', 'MAPE'])
            EvalDf = EvalDf.applymap(lambda x: '{:.3f}'.format(x))

            return EvalDf
        
        else:
            print('Tidak ada data yang diberikan untuk dilakukan evaluasi')
            return None

    def GetEvaluationIter(self, Diff = 2):
        
        ForecLists = zip(self.TrainForecastList,
                         self.TestForecastList)
        
        for order, ForecArr in zip(self.OrderList, ForecLists):
            TrainForecastArr, TestForecastArr = ForecArr

            ForecArr = zip(TrainForecastArr,
                           TestForecastArr)
            
            ActualArr = zip(self.TrainList,
                            self.TestList)

            EvalDfArr = []
            
            for i, (Actual, Forecast) in enumerate(zip(ActualArr, ForecArr)):
                Train, Test = Actual
                TrainForec, TestForec = Forecast
            
                EvalIter = self.GetEvaluation(DfTrain = Train[Diff:],
                                              DfTest = Test,
                                              DfTrainForec = TrainForec[Diff:],
                                              DfTestForec = TestForec)
                
                EvalName = f'Evaluasi_model{order}_iter{i+1}'
                locals()[EvalName] = EvalIter
                EvalDfArr.append(locals()[EvalName])
            
            self.EvalDfList.append(EvalDfArr)
        
        return self.EvalDfList

    def GetEvaluationMeanIter(self, EvalDfList = None):

        if EvalDfList is None:
            EvalDfList = self.EvalDfList
        
        self.EvalDfListMean = [(i[0].astype(float) + i[1].astype(float) + i[2].astype(float) + i[3].astype(float)) / 4 for i in EvalDfList]
        self.EvalDfListMean = [round(i, 3) for i in self.EvalDfListMean]

        return self.EvalDfListMean
    
    def GetEvaluationMeanBarPlotIter(self, padding = 5, show = 'vertical', BarWidth = 0.8, margin = 0.3):

        if show == 'vertical':
            rows, cols = 3, 1
            nHor, nVer = 2, 3
        elif show == 'horizontal':
            rows, cols = 1, 3
            nHor, nVer = 3, 2

        fig, ax = plt.subplots(nrows = rows,
                               ncols = cols,
                               figsize = (nHor * padding, 
                                          nVer * padding))
        
        MetricsList = ['RMSE (Root Mean Squared Error)',
                       'MAE (Mean Absolute Error)',
                       'MAPE (Mean Absolute Percentage Error)']
        
        for (i, idx), title in zip(enumerate(self.EvalDfListMean[0].index), MetricsList):

            EvalDfMean = [df.loc[idx] for df in self.EvalDfListMean]
            EvalDfMean = pd.concat(EvalDfMean, axis = 1)
            EvalDfMean.columns = np.arange(len(self.EvalDfListMean))

            bars = EvalDfMean.plot(kind = 'bar',
                                   ax = ax[i],
                                   edgecolor = 'black',
                                   width = BarWidth)
            
            ax[i].set_title(title,
                            fontsize = 16)
            
            ax[i].legend([f'Model ARIMA {i}' for i in self.OrderList],
                         fontsize = 8).set_visible(True)
            
            ax[i].set_xticklabels(self.EvalDfListMean[0].columns,
                                  rotation = 0)

            for j, bar in enumerate(bars.containers):
                for rect in bar:
                    height = rect.get_height()
                    ax[i].annotate(f'{height:.3f}', 
                                   xy = (rect.get_x() + rect.get_width() / 2, height),
                                   xytext = (0, 5),
                                   textcoords = "offset points",
                                   ha = 'center',
                                   va = 'bottom', 
                                   rotation = 90,
                                   fontsize = 9)
            ax[i].margins(y = margin)
        
        plt.show()
    
    def GetEvaluationBarPlotIter(self, EvalDfList = None, OrderList = None, TopMargin = 0.2, Resid = False, Comb = False, figsize = (10,10)):
        if EvalDfList is None:
            EvalDfList = self.EvalDfList
        
        if OrderList is None:
            OrderList = self.OrderList

        for order, EvalDfArr in zip(OrderList, EvalDfList):
            cols = 3
            fig, ax = plt.subplots(nrows = self.fold,
                                   ncols = cols,
                                   figsize = figsize)
            
            if Resid == True:
                fig.suptitle(f'Grafik Barplot Hasil Evaluasi Peramalan Residu Model ARIMA {order} menggunakan ANN', fontsize = 20)
            elif Comb == True:
                fig.suptitle(f'Grafik Barplot Hasil Evaluasi Kombinasi Peramalan ARIMA {order} dan ANN', fontsize = 20)
            else:
                fig.suptitle(f'Grafik Barplot Hasil Evaluasi Peramalan Model ARIMA {order}', fontsize = 20)

            for i, EvalDf in enumerate(EvalDfArr):
                EvalDf = EvalDf.astype('float')

                for j, k in enumerate(EvalDf.index):
                    EvalDf.loc[k].plot(kind = 'bar',
                                       ax = ax[i, j],
                                       color = ['blue', 'red', 'orange'])
                    ax[i, j].set_title(f'Grafik Evaluasi {k} Fold ke-{i+1}')
                    ax[i, j].set_xlabel('Metrik Evaluasi',
                                        fontsize = 14)
                    ax[i, j].set_ylabel('Nilai Evaluasi',
                                        fontsize = 14)
                    ax[i, j].set_xticklabels(EvalDf.columns, fontsize=14, rotation = 0)
                    for index, value in enumerate(EvalDf.loc[k]):
                        ax[i, j].text(index, 
                        value, 
                        str(round(value, 3)), 
                        ha='center', va='bottom')
                    
                    ax[i, j].margins(y = TopMargin)

            plt.subplots_adjust(top = 0.89,
                                wspace = 0.3,
                                hspace = 0.8)
            plt.show()
            print('\n')
    
    def GetResiduData(self, DfTrain = None, DfTest = None, DfTrainForec = None, DfTestForec = None):

        if (DfTrain is not None) and (DfTrainForec is not None):
            DfResidTrain = pd.DataFrame(DfTrain.iloc[:,0].sub(DfTrainForec.iloc[:,0],
                                                              fill_value = 0))
        
        if (DfTest is not None) and (DfTestForec is not None):
            DfResidTest = pd.DataFrame(DfTest.iloc[:,0].sub(DfTestForec.iloc[:,0],
                                                              fill_value = 0))
        
        return DfResidTrain, DfResidTest
    
    def GetResiduDataIter(self):
        
        ForecLists = zip(self.TrainForecastList,
                         self.TestForecastList)
        
        for order, ForecArr in zip(self.OrderList, ForecLists):
            TrainArr, TestArr = ForecArr

            ActualArr = zip(self.TrainList,
                            self.TestList)
            
            ForecArr = zip(TrainArr,
                           TestArr)

            TrainResiduArr = []
            TestResiduArr = []

            for i, (Actual, Forecast) in enumerate(zip(ActualArr, ForecArr)):
                Train, Test = Actual
                TrainForec, TestForec = Forecast

                DfResidTrain, DfResidTest = self.GetResiduData(DfTrain = Train,
                                                                         DfTest = Test,
                                                                         DfTrainForec = TrainForec,
                                                                         DfTestForec = TestForec)
                
                DfResidTrainName = f'DfResidTrain_Model{order}_Iter{i+1}'
                DfResidTestName  = f'DfResidTest_Model{order}_Iter{i+1}'

                locals()[DfResidTrainName] = DfResidTrain
                locals()[DfResidTestName] = DfResidTest

                TrainResiduArr.append(locals()[DfResidTrainName])
                TestResiduArr.append(locals()[DfResidTestName])

            self.DfResidTrainList.append(TrainResiduArr)
            self.DfResidTestList.append(TestResiduArr)

        return self.DfResidTrainList, self.DfResidTestList
    
    def ShowResiduSubplotIter(self, Err = 2, figsize = (22, 10)):
        
        ResidLists = zip(self.DfResidTrainList,
                         self.DfResidTestList)
        
        for order, ResidArr in zip(self.OrderList, ResidLists):
            DfResidTrainArr, DfResidTestArr = ResidArr
            
            rows = int(self.fold)
            cols = 2

            fig, ax = plt.subplots(nrows = rows,
                                   ncols = cols,
                                   figsize = figsize)
            
            fig.suptitle(f'Residu Hasil Peramalan Model ARIMA {order}',
                         fontsize = 20)
            
            for (i, (DfResidTrain, DfResidTest)) in enumerate(zip(DfResidTrainArr, DfResidTestArr)):
                ax[i, 0].plot(DfResidTrain.iloc[Err:],
                              color = 'blue')
                ax[i, 0].set_title(f'Residu Data Latih Fold ke-{i+1}')

                ax[i, 1].plot(DfResidTest,
                              color = 'blue')
                ax[i, 1].set_title(f'Residu Data uji Fold ke={i+1}')

            plt.subplots_adjust(top = 0.9,
                                hspace = 0.5,
                                wspace = 0.1)
            plt.show()
            print('\n\n')

class BPNNModel:
    def __init__(self, TrainList, TestList,
                 DfResidTrainList, DfResidTestList,
                 LookBack = 1, fold = 4, OrderList = None):
        self.LookBack = LookBack
        self.fold = fold
        self.TrainList = TrainList
        self.TestList = TestList
        self.OrderList = OrderList
        self.DfResidTrainList = DfResidTrainList
        self.DfResidTestList = DfResidTestList

        self.TrainIndex = []
        self.TestIndex = []
        self.DataIndex = []

        self.TrainScaledList = []
        self.DfTrainScaledList = []
        self.TrainScalerList = []
        self.TestScaledList = []
        self.DfTestScaledList = []
        self.TestScalerList = []

        self.XTrainList = []
        self.yTrainList = []
        self.XTestList = []
        self.yTestList = []

        self.EarlyStop = EarlyStopping(
            monitor = 'mse',
            min_delta = 0.001,
            verbose = 1,
            patience = 5,
            restore_best_weights = True,
            mode = 'min'
        )
        
        self.AMSGrad = optimizers.Adam(
            learning_rate = 0.001,
            beta_1 = 0.9,
            beta_2 = 0.999,
            epsilon = 1e-7,
            amsgrad = True,
            name = 'Adam',
            clipnorm = None,
            clipvalue = None,
            decay = 0.8,
        )

        self.model = None

        self.TrainResidForecList = []
        self.TestResidForecList = []

        self.ResidEvalDfList = []
        self.EvalDfListMean = []
    
    def SetSeed(seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)
        check_random_state(seed)

    def GetNormalizedDataIter(self):
        ResidLists = zip(self.DfResidTrainList,
                         self.DfResidTestList)

        for order, ResidArr in zip(self.OrderList, ResidLists):
            DfResidTrainArr, DfResidTestArr = ResidArr

            TrainTempArr = ([], 
                            [], 
                            [])
            
            for i, TrainResid in enumerate(DfResidTrainArr):
                TrainScaler = MinMaxScaler(feature_range = (0, 1))
                TrainScaled = TrainScaler.fit_transform(TrainResid)
                DfTrainScaled = pd.DataFrame(TrainScaled)

                TrainScalerName = f'TrainScaler_ARIMA{order}_Iter{i+1}'
                TrainScaledName = f'TrainScaled_ARIMA{order}_Iter{i+1}'
                DfTrainScaledName = f'DfTrainScaled_ARIMA{order}_Iter{i+1}'

                locals()[TrainScalerName] = TrainScaler
                locals()[TrainScaledName] = TrainScaled
                locals()[DfTrainScaledName] = DfTrainScaled

                TrainTempArr[0].append(locals()[TrainScalerName])
                TrainTempArr[1].append(locals()[TrainScaledName])
                TrainTempArr[2].append(locals()[DfTrainScaledName])
            
            self.TrainScalerList.append(TrainTempArr[0])
            self.TrainScaledList.append(TrainTempArr[1])
            self.DfTrainScaledList.append(TrainTempArr[2])

            TestTempArr = ([], 
                            [], 
                            [])
            
            for i, TestResid in enumerate(DfResidTestArr):
                TestScaler = MinMaxScaler(feature_range = (0, 1))
                TestScaled = TestScaler.fit_transform(TestResid)
                DfTestScaled = pd.DataFrame(TestScaled)

                TestScalerName = f'TestScaler_ARIMA{order}_Iter{i+1}'
                TestScaledName = f'TestScaled_ARIMA{order}_Iter{i+1}'
                DfTestScaledName = f'DfTestScaled_ARIMA{order}_Iter{i+1}'

                locals()[TestScalerName] = TestScaler
                locals()[TestScaledName] = TestScaled
                locals()[DfTestScaledName] = DfTestScaled

                TestTempArr[0].append(locals()[TestScalerName])
                TestTempArr[1].append(locals()[TestScaledName])
                TestTempArr[2].append(locals()[DfTestScaledName])
            
            self.TestScalerList.append(TestTempArr[0])
            self.TestScaledList.append(TestTempArr[1])
            self.DfTestScaledList.append(TestTempArr[2])

        return (self.TrainScalerList, self.TrainScaledList, self.DfTrainScaledList,
                self.TestScalerList, self.TestScaledList, self.DfTestScaledList)
    
    def GetIndex(self):

        for i, TrainIdx in enumerate(self.TrainList):
            TrainIdx = TrainIdx.index
            TrainIdxName = f'TrainIdx{i+1}'
            locals()[TrainIdxName] = TrainIdx
            self.TrainIndex.append(locals()[TrainIdxName])

        for i, TestIdx in enumerate(self.TestList):
            TestIdx = TestIdx.index
            TestIdxName = f'TestIdx{i+1}'
            locals()[TestIdxName] = TestIdx
            self.TestIndex.append(locals()[TestIdxName])
        
        return self.TrainIndex, self.TestIndex

    def SlidingWindow(self, Df):
        X, y = [], []
        for i in range(len(Df)- self.LookBack - 1):
            X.append(Df[i:(i + self.LookBack) , 0])
            y.append(Df[i + self.LookBack, 0])
        
        return np.array(X), np.array(y)

    def SlidingWindowIter(self):
        ScaledList = zip(self.TrainScaledList,
                         self.TestScaledList)
        
        for order, ScaledArr in zip(self.OrderList, ScaledList):
            TrainScaledArr, TestScaledArr = ScaledArr

            TrainTempArr = ([],
                            [])
            
            for i, TrainScaled in enumerate(TrainScaledArr):
                XTrain, yTrain = self.SlidingWindow(TrainScaled)
                XTrainName = f'XTrain_ARIMA{order}_iter{i+1}'
                yTrainName = f'yTrain_ARIMA{order}_iter{i+1}'

                locals()[XTrainName] = XTrain
                locals()[yTrainName] = yTrain

                TrainTempArr[0].append(locals()[XTrainName])
                TrainTempArr[1].append(locals()[yTrainName])
            
            self.XTrainList.append(TrainTempArr[0])
            self.yTrainList.append(TrainTempArr[1])

            TestTempArr = ([],
                            [])
            
            for i, TestScaled in enumerate(TestScaledArr):
                XTest, yTest = self.SlidingWindow(TestScaled)
                XTestName = f'XTest_ARIMA{order}_iter{i+1}'
                yTestName = f'yTest_ARIMA{order}_iter{i+1}'

                locals()[XTestName] = XTest
                locals()[yTestName] = yTest

                TestTempArr[0].append(locals()[XTestName])
                TestTempArr[1].append(locals()[yTestName])
            
            self.XTestList.append(TestTempArr[0])
            self.yTestList.append(TestTempArr[1])
        
        return (self.XTrainList, self.yTrainList,
                self.XTestList, self.yTestList)
    
    def MakeModel(self, InputNeuron:int = 1, OutputNeuron:int = 1, ShowSummary = False, PrintPlot = False, Opt = 'amsgrad'):
        if Opt == 'amsgrad':
            Opt = self.AMSGrad
        else:
            Opt = Opt
        
        self.model = Sequential()

        self.model.add(Dense(InputNeuron,
                             input_shape = (self.LookBack,), 
                             activation = 'relu'))
        
        self.model.add(Dense(OutputNeuron,
                             activation = 'relu'))
        
        self.model.compile(loss = 'mean_squared_error',
                           metrics = 'mse',
                           optimizer = Opt)
        
        if ShowSummary == True:
            self.model.summary()

        if PrintPlot == True:
            plot_model(self.model,
                       to_file = 'BPNNModelPlot.png',
                       show_shapes = True,
                       show_layer_names = True)
        
        return self.model

    def ResidForecastIter(self, model = None, Err = 0, MaxEpoch = 100):
        start_time = time.time()

        if model is not None:
            model = self.model

        SlideWindowList = zip(self.XTrainList,
                              self.yTrainList,
                              self.XTestList,
                              self.yTestList)
        
        ScalerList = zip(self.TrainScalerList,
                         self.TestScalerList)
        
        for order, ScalerArr, SlideWindowArr in zip(self.OrderList, ScalerList, SlideWindowList):
            TrainResidForecArr, TestResidForecArr = [], []

            XTrainArr, yTrainArr, XTestArr, yTestArr = SlideWindowArr
            SlideWindowArr = zip(XTrainArr, yTrainArr, 
                                 XTestArr, yTestArr)
            
            TrainScalerArr, TestScalerArr = ScalerArr
            ScalerArr = zip(TrainScalerArr,
                            TestScalerArr)
            IndexArr = zip(self.TrainIndex,
                           self.TestIndex)
            
            for Index, Scaler, Xy in zip(IndexArr, ScalerArr, SlideWindowArr):
                TrainIndex, TestIndex = Index
                TrainScaler, TestScaler = Scaler
                XTrain, yTrain, XTest, yTest = Xy

                model.fit(XTrain, yTrain,
                          epochs = MaxEpoch,
                          validation_data = (XTrain, yTrain),
                          shuffle = False,
                          callbacks = [self.EarlyStop])
                
                model.evaluate(XTest, yTest,
                               verbose = 0)
                
                TrainResidForec = model.predict(XTrain)
                TrainResidForec = TrainScaler.inverse_transform(TrainResidForec)
                TrainResidForec = pd.DataFrame(TrainResidForec,
                                               index = TrainIndex[self.LookBack + Err + 1:])
                TrainResidForec.columns = ['Peramalan']

                TrainResidForecName = f'TrainResidForec_{order}_Iter{i+1}'
                locals()[TrainResidForecName] = TrainResidForec
                TrainResidForecArr.append(locals()[TrainResidForecName])

                TestResidForec = model.predict(XTest)
                TestResidForec = TestScaler.inverse_transform(TestResidForec)
                TestResidForec = pd.DataFrame(TestResidForec,
                                               index = TestIndex[self.LookBack + Err + 1:])
                TestResidForec.columns = ['Peramalan']

                TestResidForecName = f'TestResidForec_{order}_Iter{i+1}'
                locals()[TestResidForecName] = TestResidForec
                TestResidForecArr.append(locals()[TestResidForecName])

            self.TrainResidForecList.append(TrainResidForecArr)
            self.TestResidForecList.append(TestResidForecArr)
        
        elapsed_time = time.time() - start_time
        print('\nTime: {:.2f} seconds'.format(elapsed_time))

        return self.TrainResidForecList, self.TestResidForecList
    
    def ShowResidSubplotIter(self, Err = 0, LabelActual = 'Data Residu', LabelForec = 'Data Peramalan', figsize = (18,10),
                        DfTrainList = None, DfTestList = None, DfTrainForecList = None, DfTestForecList = None):
        
        if Err == 0:
            Err = self.LookBack + 2

        if (DfTrainList is None) and (DfTestList is None) and (DfTrainForecList is None) and (DfTestForecList is None):
        # if not any((DfDataList, DfTrainList, DfTestList, DfDataForecList, DfTrainForecList, DfTestForecList)):
            DfTrainList = self.DfResidTrainList
            DfTestList = self.DfResidTestList
            DfTrainForecList = self.TrainResidForecList
            DfTestForecList = self.TestResidForecList
        

        ResidForecastLists = zip(DfTrainForecList,
                                 DfTestForecList)
        
        DfResidLists = zip(DfTrainList,
                           DfTestList)
        
        rows = int(self.fold )

        LabelActual = LabelActual
        LabelForec = LabelForec
        
        for order, ResidForecastArr, ResidArr in zip(self.OrderList, ResidForecastLists, DfResidLists):
            TrainResidForecastArr, TestResidForecastArr = ResidForecastArr
            TrainResidArr, TestResidArr = ResidArr

            fig, ax = plt.subplots(nrows = rows,
                                   ncols = 2,
                                   figsize = figsize)
            
            fig.suptitle(f'Peramalan Data Residu Model ARIMA {order}', fontsize = 18)
            
            for i in range(self.fold):
                ax[i, 0].plot(TrainResidArr[i].iloc[Err:], 
                              label = LabelActual, 
                              color = 'blue')
                ax[i, 0].plot(TrainResidForecastArr[i].iloc[Err:],
                              label = LabelForec,
                            color = 'red')
                ax[i, 0].set_title(f'Peramalan Data Latih Fold ke-{i+1}')

                ax[i, 1].plot(TestResidArr[i],
                              label = LabelActual,
                              color = 'blue')
                ax[i, 1].plot(TestResidForecastArr[i],
                              label = LabelForec,
                              color = 'red')
                ax[i, 1].set_title(f'Peramalan Data Uji Fold ke-{i+1}')
            
            plt.subplots_adjust(top = 0.9,
                                hspace = 0.5,
                                wspace = 0.1)
            plt.show()
            print('\n\n')

    def GetResidEvaluationIter(self, Err = 2, AddErr = 1):
        
        LookBack = self.LookBack
        
        ResidForecastLists = zip(self.TrainResidForecList,
                                 self.TestResidForecList)
        
        DfResidLists = zip(self.DfResidTrainList,
                           self.DfResidTestList)
        
        for order, ForecArr, ResidArr in zip(self.OrderList, ResidForecastLists, DfResidLists):
            TrainResidForecArr, TestResidForecArr = ForecArr

            ForecArr = zip(TrainResidForecArr,
                           TestResidForecArr)
            
            ResidTrainArr, ResidTestArr = ResidArr

            ResidArr = zip(ResidTrainArr,
                           ResidTestArr)

            ResidEvalDfArr = []
            
            for i, (Resid, Forecast) in enumerate(zip(ResidArr, ForecArr)):
                TrainResid, TestResid = Resid
                TrainResidForec, TestResidForec = Forecast
            
                EvalIter = ARIMAModel.GetEvaluation(self = self,
                                                    DfTrain = TrainResid[LookBack + Err + AddErr:],
                                                    DfTest = TestResid[LookBack + Err + AddErr:],
                                                    DfTrainForec = TrainResidForec[Err:],
                                                    DfTestForec = TestResidForec[Err:])
                
                EvalName = f'Evaluasi_model{order}_iter{i+1}'
                locals()[EvalName] = EvalIter
                ResidEvalDfArr.append(locals()[EvalName])
            
            self.ResidEvalDfList.append(ResidEvalDfArr)
        
        return self.ResidEvalDfList
    

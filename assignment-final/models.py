import pandas
import time
import datetime
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt

train_features = pandas.read_csv('./features.csv', index_col='match_id')

count_nan_train = len(train_features) - train_features.count()
count_nan_train = count_nan_train[count_nan_train != 0]

print( count_nan_train )

train_features.drop(['duration', 
         'tower_status_radiant', 
         'tower_status_dire', 
         'barracks_status_radiant', 
         'barracks_status_dire'
        ], axis=1, inplace=True)
		
X_train = train_features.fillna(0)
y_train = train_features['radiant_win']

del X_train['radiant_win']

trees = [10,20,30,40,100,200,250]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = []

for n in trees:
	start_time = datetime.datetime.now()
	clf = GradientBoostingClassifier(n_estimators=n, random_state=42)
	print ( "n = " + str(n), end=' ' )
	cva = cross_val_score(clf, X_train, y_train, cv=kf, scoring='roc_auc')
	print( cva, end=' ' )
	print('mean ' + str( round(np.mean(cva),2 )), end= ' ')
	print('Time elapsed:', datetime.datetime.now() - start_time)
	scores.append(np.mean(cva))

plt.figure()
plt.plot(trees,scores)
plt.xlabel('n_estimators')
plt.ylabel('score')
plt.savefig('gbc.png')

# first_blood_time               19553
# first_blood_team               19553
# first_blood_player1            19553
# first_blood_player2            43987
# radiant_bottle_time            15691
# radiant_courier_time             692
# radiant_flying_courier_time    27479
# radiant_first_ward_time         1836
# dire_bottle_time               16143
# dire_courier_time                676
# dire_flying_courier_time       26098
# dire_first_ward_time            1826
# dtype: int64
# n = 10 [ 0.66383799  0.66635457  0.66360048  0.66529818  0.66516222] mean 0.66 Time elapsed: 0:00:28.059678
# n = 20 [ 0.68083889  0.68272733  0.67969876  0.6834932   0.6855512 ] mean 0.68 Time elapsed: 0:01:12.938698
# n = 30 [ 0.68892093  0.68934663  0.68712298  0.69180598  0.69283583] mean 0.69 Time elapsed: 0:02:06.911668
# n = 40 [ 0.69264125  0.69335305  0.69153074  0.69586466  0.69680392] mean 0.69 Time elapsed: 0:03:15.468382
# n = 100 [ 0.70515496  0.706077    0.70429951  0.7074682   0.70811523] mean 0.71 Time elapsed: 0:06:36.796986
# n = 200 [ 0.71214175  0.71304181  0.71259725  0.71556107  0.71510233] mean 0.71 Time elapsed: 0:08:20.422425
# n = 250 [ 0.71547078  0.71462661  0.71535522  0.71731507  0.71680663] mean 0.72 Time elapsed: 0:10:25.212556

#--------------------------------------
def find_C(X, y, name):
	scores = []
	C_range = list(range(-5,6))
	for C in [ 10 ** i for i in C_range]:
		start_time = datetime.datetime.now()
		clf = LogisticRegression(C=C, random_state=42)
		print ( "C = " + str(C), end=' ' )
		cva = cross_val_score(clf, X, y, cv=kf, scoring='roc_auc')
		print( cva, end=' ' )
		scores.append(np.mean(cva))
		print('mean ' + str( round(np.mean(cva),2 )), end= ' ')
		print('Time elapsed:', datetime.datetime.now() - start_time)
	max_score = max(scores)
	max_score_index = scores.index(max_score)
	plt.figure()
	plt.plot(C_range,scores)
	plt.xlabel('C_range')
	plt.ylabel('score')
	plt.savefig('lr_' + name + '.png')	
	return 10 ** C_range[max_score_index], max_score
#--------------------------------------
def heroes_bag(data,N):
	X_pick = np.zeros((data.shape[0], N))
	for i, match_id in enumerate(data.index):
		for p in range(5):
			X_pick[i, data.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
			X_pick[i, data.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1
	return pandas.DataFrame(X_pick, index=data.index)
#--------------------------------------
def test_model(X,y,image_name):
	(C, max_score) = find_C(X, y, image_name)
	print("max C = {} and max_score = {}".format(C,max_score))
#--------------------------------------
def clean(X):
	X.drop(['duration', 
         'tower_status_radiant', 
         'tower_status_dire', 
         'barracks_status_radiant', 
         'barracks_status_dire'
        ], axis=1, inplace=True)
	y = X['radiant_win']
	del X['radiant_win']
	return X.fillna(0), y
#--------------------------------------
def clean_cat(X):
	X.drop(['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero','r5_hero','d1_hero', 'd2_hero', 'd3_hero', 'd4_hero','d5_hero'], axis=1, inplace=True)
	return X.fillna(0)
#--------------------------------------
	
train = pandas.read_csv('./features.csv', index_col='match_id')
test = pandas.read_csv('./features_test.csv', index_col='match_id')

scaler = StandardScaler()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

(X_train, y_train) = clean(train)
X_train_scalered = scaler.fit_transform(X_train)

print("1. Use all features")

test_model(X_train_scalered, y_train, '1')

heroes = pandas.read_csv('./data/dictionaries/heroes.csv')
N = len(heroes)
print("- All heroes in game: {}".format(N))
X_heroes = heroes_bag(X_train, N)

X_train = clean_cat(X_train)
X_train_scalered = scaler.fit_transform(X_train)

print("2. After delete categorial features")

test_model(X_train_scalered,y_train,'2')

X_train = pandas.DataFrame(scaler.fit_transform(X_train), index = X_train.index)
X_train = pandas.concat([X_train, X_heroes], axis=1)

print("3. After words bag")

test_model(X_train,y_train,'3')

print("4. Predict on test data")

model = LogisticRegression(C=0.1, random_state=42)
model.fit(X_train, y_train)

X_heroes = heroes_bag(test, N)
X_test = clean_cat(test)

X_test = pandas.DataFrame(scaler.fit_transform(X_test), index = X_test.index)
X_test = pandas.concat([X_test, X_heroes], axis=1)
pred = model.predict_proba(X_test)[:, 1]

print("max pred: {} min pred: {}".format( pred.max(), pred.min()))

# 1. Use all features
# C = 1e-05 [ 0.6931532   0.69481539  0.69571379  0.69513159  0.69699445] mean 0.7 Time elapsed: 0:00:04.661957
# C = 0.0001 [ 0.70956686  0.71039474  0.71170336  0.71176082  0.71336295] mean 0.71 Time elapsed: 0:00:07.145306
# C = 0.001 [ 0.71449541  0.71577214  0.71625974  0.71697301  0.71831738] mean 0.72 Time elapsed: 0:00:13.764188
# C = 0.01 [ 0.71464598  0.71617915  0.71624749  0.71735041  0.71832832] mean 0.72 Time elapsed: 0:00:16.131662
# C = 0.1 [ 0.71462192  0.71617479  0.71619187  0.71737596  0.7182712 ] mean 0.72 Time elapsed: 0:00:17.525726
# C = 1 [ 0.71461815  0.71617185  0.71618338  0.7173762   0.71826341] mean 0.72 Time elapsed: 0:00:17.678581
# C = 10 [ 0.71461695  0.71617179  0.71618281  0.71737669  0.7182632 ] mean 0.72 Time elapsed: 0:00:17.522182
# C = 100 [ 0.71461692  0.71617206  0.71618249  0.71737655  0.71826345] mean 0.72 Time elapsed: 0:00:18.396353
# C = 1000 [ 0.71461693  0.71617214  0.7161825   0.71737659  0.71826346] mean 0.72 Time elapsed: 0:00:17.639451
# C = 10000 [ 0.71461693  0.71617215  0.71618251  0.7173766   0.71826345] mean 0.72 Time elapsed: 0:00:17.639299
# C = 100000 [ 0.71461693  0.71617215  0.71618252  0.71737661  0.71826345] mean 0.72 Time elapsed: 0:00:17.865554
# max C = 0.01 and max_score = 0.7165502697259141
# - All heroes in game: 112
# 2. After delete categorial features
# C = 1e-05 [ 0.69301063  0.69476367  0.69562225  0.69507488  0.69702952] mean 0.7 Time elapsed: 0:00:04.275846
# C = 0.0001 [ 0.70936502  0.71042199  0.71168956  0.71168269  0.71353648] mean 0.71 Time elapsed: 0:00:06.710933
# C = 0.001 [ 0.71434804  0.71581532  0.71629769  0.7168852   0.71853275] mean 0.72 Time elapsed: 0:00:11.826886
# C = 0.01 [ 0.71450425  0.71622052  0.71627999  0.71725088  0.71854131] mean 0.72 Time elapsed: 0:00:14.983336
# C = 0.1 [ 0.71448206  0.71620992  0.71622569  0.71726917  0.71848436] mean 0.72 Time elapsed: 0:00:16.644224
# C = 1 [ 0.71447768  0.71620627  0.7162209   0.71727019  0.71847678] mean 0.72 Time elapsed: 0:00:17.107220
# C = 10 [ 0.71447669  0.71620645  0.71621968  0.71727123  0.71847581] mean 0.72 Time elapsed: 0:00:17.138976
# C = 100 [ 0.7144767   0.71620624  0.71621963  0.71727119  0.71847569] mean 0.72 Time elapsed: 0:00:18.715633
# C = 1000 [ 0.71447667  0.71620624  0.71621965  0.71727119  0.71847569] mean 0.72 Time elapsed: 0:00:18.795073
# C = 10000 [ 0.71447668  0.71620627  0.71621961  0.71727117  0.71847569] mean 0.72 Time elapsed: 0:00:18.306526
# C = 100000 [ 0.71447668  0.71620627  0.71621961  0.71727117  0.71847569] mean 0.72 Time elapsed: 0:00:18.823987
# max C = 0.01 and max_score = 0.7165593885630225
# 3. After words bag
# C = 1e-05 [ 0.69720207  0.69884021  0.69957686  0.69934071  0.70112123] mean 0.7 Time elapsed: 0:00:05.299830
# C = 0.0001 [ 0.72359118  0.72409682  0.72469067  0.72599494  0.72698389] mean 0.73 Time elapsed: 0:00:07.942378
# C = 0.001 [ 0.74465332  0.74667772  0.74413152  0.74880233  0.74740586] mean 0.75 Time elapsed: 0:00:15.761002
# C = 0.01 [ 0.74947407  0.75279102  0.7492512   0.75538896  0.75178335] mean 0.75 Time elapsed: 0:00:25.792932
# C = 0.1 [ 0.74943218  0.75315601  0.74950643  0.7559819   0.75166068] mean 0.75 Time elapsed: 0:00:36.468265
# C = 1 [ 0.74936295  0.75316822  0.74949775  0.75602112  0.75158796] mean 0.75 Time elapsed: 0:00:38.416071
# C = 10 [ 0.74935121  0.75317095  0.74949831  0.75602517  0.75158   ] mean 0.75 Time elapsed: 0:00:37.000929
# C = 100 [ 0.74935011  0.75317056  0.74949869  0.75602547  0.75157877] mean 0.75 Time elapsed: 0:00:36.918180
# C = 1000 [ 0.74934946  0.75317052  0.74949877  0.75602565  0.75157867] mean 0.75 Time elapsed: 0:00:37.461169
# C = 10000 [ 0.74934982  0.75317055  0.74949872  0.75602524  0.7515786 ] mean 0.75 Time elapsed: 0:00:37.407246
# C = 100000 [ 0.74935058  0.7531704   0.74949892  0.75602535  0.7515787 ] mean 0.75 Time elapsed: 0:00:37.018587
# max C = 0.1 and max_score = 0.7519474413465284
# max pred: 0.9964586771296121 min pred: 0.008580592559245431
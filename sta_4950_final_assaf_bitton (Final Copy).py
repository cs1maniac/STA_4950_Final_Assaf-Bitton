!pip install -q yfinance

import numpy as np
import pandas as pd
import yfinance as yf
from google.colab import files
import zipfile
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

sdate="2014-01-01"
edate="2026-05-20"

tagbag={
"AAPL":"Tech","MSFT":"Tech","GOOGL":"Tech","META":"Tech","NVDA":"Tech","ORCL":"Tech","CRM":"Tech","ADBE":"Tech","AMD":"Tech","INTC":"Tech","IBM":"Tech","CSCO":"Tech","QCOM":"Tech",
"JPM":"Financials","BAC":"Financials","WFC":"Financials","GS":"Financials","MS":"Financials","C":"Financials","AXP":"Financials","BLK":"Financials","SCHW":"Financials","USB":"Financials","PNC":"Financials","BK":"Financials",
"XOM":"Energy","CVX":"Energy","COP":"Energy","SLB":"Energy","EOG":"Energy","PSX":"Energy","VLO":"Energy","MPC":"Energy","OXY":"Energy","HAL":"Energy",
"JNJ":"Health","PFE":"Health","MRK":"Health","UNH":"Health","ABBV":"Health","TMO":"Health","BMY":"Health","AMGN":"Health","GILD":"Health","ABT":"Health","MDT":"Health","CVS":"Health",
"PG":"Staples","KO":"Staples","PEP":"Staples","WMT":"Staples","COST":"Staples","CL":"Staples","MDLZ":"Staples","MO":"Staples","PM":"Staples","KMB":"Staples",
"AMZN":"Discretionary","TSLA":"Discretionary","HD":"Discretionary","MCD":"Discretionary","NKE":"Discretionary","SBUX":"Discretionary","LOW":"Discretionary","TJX":"Discretionary","BKNG":"Discretionary","TGT":"Discretionary",
"BA":"Industrials","CAT":"Industrials","HON":"Industrials","UPS":"Industrials","GE":"Industrials","DE":"Industrials","LMT":"Industrials","RTX":"Industrials","MMM":"Industrials","UNP":"Industrials",
"NEE":"Utilities","DUK":"Utilities","SO":"Utilities","AEP":"Utilities","EXC":"Utilities","XEL":"Utilities","SRE":"Utilities","D":"Utilities","PEG":"Utilities","WEC":"Utilities",
"LIN":"Materials","APD":"Materials","SHW":"Materials","ECL":"Materials","NEM":"Materials","FCX":"Materials","DOW":"Materials",
"PLD":"RealEstate","AMT":"RealEstate","CCI":"RealEstate","SPG":"RealEstate","O":"RealEstate",
"T":"Communication","VZ":"Communication","CMCSA":"Communication","DIS":"Communication","NFLX":"Communication"
}

rawpx=yf.download(list(tagbag.keys()),start=sdate,end=edate,auto_adjust=True,progress=False)["Close"]

keepers=[z for z in tagbag.keys() if z in rawpx.columns and rawpx[z].notna().mean()>=0.95]
pxbox=rawpx[keepers].dropna()

if pxbox.shape[1]<100:
    raise ValueError(f"Only {pxbox.shape[1]} stocks survived. Need at least 100.")

pxbox=pxbox.iloc[:,:100]
sectbox=pd.Series({z:tagbag[z] for z in pxbox.columns},name="sector")

pxbox.to_csv("equity_close_prices.csv")
sectbox.to_csv("equity_sector_labels.csv")

with zipfile.ZipFile("equity_data_files.zip","w") as zippy:
    zippy.write("equity_close_prices.csv")
    zippy.write("equity_sector_labels.csv")

print(pxbox.shape)
print(pxbox.index.min().date(),pxbox.index.max().date())

files.download("equity_data_files.zip")

import numpy as np
import pandas as pd
from google.colab import files

pxbox=pd.read_csv("equity_close_prices.csv",index_col=0,parse_dates=True)

retbox=np.log(pxbox/pxbox.shift(1)).dropna()

retbox.to_csv("equity_log_returns.csv")

print(retbox.shape)
print(retbox.index.min().date(),retbox.index.max().date())
print(retbox.iloc[:5,:5])

files.download("equity_log_returns.csv")

!pip install -q yfinance

import numpy as np
import pandas as pd
import yfinance as yf
from google.colab import files
import warnings
warnings.filterwarnings("ignore")

retbox=pd.read_csv("equity_log_returns.csv",index_col=0,parse_dates=True)

rfraw=yf.download("^IRX",start=retbox.index.min(),end=None,progress=False)["Close"]

if isinstance(rfraw,pd.DataFrame):
    rfraw=rfraw.iloc[:,0]

ybox=(rfraw/100).reindex(retbox.index).ffill()
rfbox=np.log(1+ybox)/252

rfbox.name="rf_daily_log_return"
rfbox.to_csv("riskfree_daily_log_returns.csv")

print(rfbox.shape)
print(rfbox.index.min().date(),rfbox.index.max().date())
print(rfbox.head())

files.download("riskfree_daily_log_returns.csv")

import numpy as np
import pandas as pd
from google.colab import files

retbox=pd.read_csv("equity_log_returns.csv",index_col=0,parse_dates=True)
rfbox=pd.read_csv("riskfree_daily_log_returns.csv",index_col=0,parse_dates=True)

rfline=rfbox.iloc[:,0].reindex(retbox.index).ffill()

xretbox=retbox.subtract(rfline,axis=0)

xretbox.to_csv("equity_excess_log_returns.csv")

print(xretbox.shape)
print(xretbox.index.min().date(),xretbox.index.max().date())
print(xretbox.iloc[:5,:5])

files.download("equity_excess_log_returns.csv")

import numpy as np
import pandas as pd
from google.colab import files

xretbox=pd.read_csv("equity_excess_log_returns.csv",index_col=0,parse_dates=True)
sectbox=pd.read_csv("equity_sector_labels.csv",index_col=0)

X=xretbox.T

sectbox=sectbox.reindex(X.index)

X.to_csv("X_equity_excess_matrix.csv")

print(X.shape)
print("stocks =",X.shape[0])
print("trading days =",X.shape[1])
print(X.iloc[:5,:5])

files.download("X_equity_excess_matrix.csv")

import numpy as np
import pandas as pd

X=pd.read_csv("X_equity_excess_matrix.csv",index_col=0)

U,s,Vt=np.linalg.svd(X.values,full_matrices=False)

U3=U[:,:3]
s3=s[:3]
Vt3=Vt[:3,:]

vshare=(s**2)/(s**2).sum()
top3=vshare[:3].sum()

svtab=pd.DataFrame({
    "component":["PC1","PC2","PC3"],
    "singular_value":s3,
    "variance_fraction":vshare[:3],
    "cumulative_fraction":np.cumsum(vshare[:3])
})

print(svtab.round(6))
print("Top 3 fraction =",round(top3,6))
print("Top 3 percent =",round(100*top3,2))

"""Using the excess-return matrix X without centering or standardizing, the first three SVD components capture 49.65% of total variability. Individually, PC1 captures 35.49%, PC2 captures 7.81%, and PC3 captures 6.35%."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files

X=pd.read_csv("X_equity_excess_matrix.csv",index_col=0)
days=pd.to_datetime(X.columns)

U,s,Vt=np.linalg.svd(X.values,full_matrices=False)

tonebag=["crimson","darkgreen","purple"]
mainbag=[
    ("PC1 temporal component","COVID market crash","2020-03-16"),
    ("PC2 temporal component","Vaccine / market rotation","2020-11-09"),
    ("PC3 temporal component","Tariff-policy volatility","2025-04-03")
]

fig,axes=plt.subplots(3,1,figsize=(13,8),sharex=True)

for k,ax in enumerate(axes):
    title,lab,dt=mainbag[k]
    dtx=pd.Timestamp(dt)
    ax.plot(days,Vt[k,:],color=tonebag[k],lw=1.1)
    ax.axhline(0,color="black",lw=0.6)
    ax.axvline(dtx,color="black",ls="--",lw=1.0,alpha=0.75)
    ax.text(dtx,ax.get_ylim()[1],lab,rotation=90,va="top",ha="right",fontsize=9)
    ax.set_title(title,loc="left")
    ax.set_ylabel(f"v{k+1}(t)")
    ax.grid(alpha=0.25)

axes[-1].set_xlabel("Date")
plt.suptitle("Three SVD temporal components with one representative event per component",y=1.02)
plt.tight_layout()
plt.savefig("svd_temporal_components_one_event_each.png",dpi=150,bbox_inches="tight")
plt.show()

for k in range(3):
    tmp=pd.Series(Vt[k,:],index=days)
    print(f"\nLargest absolute dates for PC{k+1}:")
    print(tmp.abs().sort_values(ascending=False).head(8))

files.download("svd_temporal_components_one_event_each.png")

"""The temporal components show event-driven spikes rather than smooth long-run trends. PC1 is dominated by the March 2020 COVID market crash, with its largest absolute value on March 16, 2020. PC2 has its largest spike on November 9, 2020, which lines up with the vaccine-news market rotation. PC3 is also affected by the March 2020 volatility, but it has later visible spikes in 2025; I mark April 2025 as a representative trade-policy shock. Overall, the components do track recognizable macro/geopolitical or market-wide events, although not perfectly. The sign of each component is arbitrary, so the timing and magnitude of spikes matter more than whether they are positive or negative."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files

X=pd.read_csv("X_equity_excess_matrix.csv",index_col=0)
sectbox=pd.read_csv("equity_sector_labels.csv",index_col=0)

if isinstance(sectbox,pd.DataFrame):
    sectline=sectbox.iloc[:,0]
else:
    sectline=sectbox

sectline=sectline.reindex(X.index)

U,s,Vt=np.linalg.svd(X.values,full_matrices=False)

spotbox=X.values@Vt[:3,:].T

projbox=pd.DataFrame(spotbox,index=X.index,columns=["SVD1","SVD2","SVD3"])
projbox["sector"]=sectline.values

fig=plt.figure(figsize=(11,8))
ax=fig.add_subplot(111,projection="3d")

sectors=sorted(projbox["sector"].dropna().unique())
tonebag=dict(zip(sectors,plt.cm.tab20(np.linspace(0,1,len(sectors)))))

for sec in sectors:
    tmp=projbox[projbox["sector"]==sec]
    ax.scatter(tmp["SVD1"],tmp["SVD2"],tmp["SVD3"],s=55,color=tonebag[sec],label=sec,edgecolor="black",linewidth=0.35,alpha=0.85)

ax.set_xlabel("SVD component 1")
ax.set_ylabel("SVD component 2")
ax.set_zlabel("SVD component 3")
ax.set_title("3D stock projections onto first three SVD components")
ax.view_init(elev=22,azim=40)
ax.legend(fontsize=8,loc="upper left",bbox_to_anchor=(1.02,1))

plt.tight_layout()
plt.savefig("svd_stock_projection_3d.png",dpi=160,bbox_inches="tight")
plt.show()

files.download("svd_stock_projection_3d.png")

"""The 3D SVD projection shows partial alignment with industry membership. Energy stocks form one of the clearest separate clusters, and Tech stocks also tend to occupy a similar region, with NVDA appearing more extreme than the rest of the group. Financials and Industrials are more centralized, while Staples, Utilities, and Health are closer together in the lower part of the plot. The clusters are not perfectly separated because the leading SVD components still capture broad market-wide variation, not only sector-specific behavior. Overall, industry membership is visible, but only partially."""

!pip install -q umap-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from google.colab import files
import warnings
warnings.filterwarnings("ignore")

files.upload()

X=pd.read_csv("X_equity_excess_matrix.csv",index_col=0)
sectbox=pd.read_csv("equity_sector_labels.csv",index_col=0)
sectline=sectbox.iloc[:,0].reindex(X.index)

U,s,Vt=np.linalg.svd(X.values,full_matrices=False)
emb_svd=X.values@Vt[:3,:].T

perp=10
nnear=10
mind=.10
seed=42

emb_tsne=TSNE(n_components=3,perplexity=perp,init="pca",learning_rate="auto",random_state=seed).fit_transform(X.values)

umod=umap.UMAP(n_components=3,n_neighbors=nnear,min_dist=mind,metric="euclidean",random_state=seed)
emb_umap=umod.fit_transform(X.values)

plotbag=[
    (emb_svd,"SVD projection"),
    (emb_tsne,f"t-SNE 3D, perplexity={perp}"),
    (emb_umap,f"UMAP 3D, neighbors={nnear}, min_dist={mind}")
]

sectors=sorted(sectline.dropna().unique())
tonebag=dict(zip(sectors,plt.cm.tab20(np.linspace(0,1,len(sectors)))))

fig=plt.figure(figsize=(9,18))

for j,(emb,title) in enumerate(plotbag,1):
    ax=fig.add_subplot(3,1,j,projection="3d")
    for sec in sectors:
        mask=(sectline==sec).values
        ax.scatter(emb[mask,0],emb[mask,1],emb[mask,2],s=55,color=tonebag[sec],edgecolor="black",linewidth=.25,alpha=.85,label=sec if j==1 else None)
    ax.set_title(title,fontsize=12)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_zlabel("dim 3")
    ax.view_init(elev=22,azim=40)

fig.legend(loc="lower center",ncol=5,fontsize=8,bbox_to_anchor=(.5,.02))
plt.suptitle("Same 100 stocks embedded three ways; colors show sector labels",y=.995)
plt.tight_layout(rect=[0,.045,1,.985])
plt.savefig("svd_tsne_umap_3d_comparison_vertical.png",dpi=180,bbox_inches="tight")
plt.show()

print("t-SNE perplexity =",perp)
print("UMAP n_neighbors =",nnear)
print("UMAP min_dist =",mind)

files.download("svd_tsne_umap_3d_comparison_vertical.png")

"""I used one 3D embedding for each method so the three representations could be compared directly. For t-SNE, I used perplexity = 10 because the sample has only 100 stocks, so a smaller perplexity focuses on local neighborhoods without requiring too many neighbors per point. For UMAP, I used n_neighbors = 10 for a similar local-neighborhood scale, and min_dist = 0.10 to allow moderately tight clusters without forcing points to collapse.

The SVD plot shows partial sector structure, especially for Energy and Tech, but with overlap across sectors. The t-SNE embedding is less cleanly sector-separated, suggesting that local return similarities do not map perfectly onto industry labels. UMAP shows the clearest clustering, with several sectors forming tighter groups. Overall, the nonlinear embeddings reveal more local structure than SVD, but sector membership still explains only part of the return behavior.
"""

import numpy as np
import pandas as pd
import requests,time
from google.colab import files

cstart="2016-01-01"
cend="2026-05-19"

def tsnum(x):
    return int(pd.Timestamp(x,tz="UTC").timestamp())

def ccgrab(sym,start,end):
    start_ts=tsnum(start)
    hit_ts=tsnum(end)
    chunks=[]

    while True:
        url="https://min-api.cryptocompare.com/data/v2/histoday"
        pars={"fsym":sym,"tsym":"USD","limit":2000,"toTs":hit_ts}
        out=requests.get(url,params=pars,timeout=30).json()

        if out.get("Response")!="Success":
            raise RuntimeError(out)

        z=pd.DataFrame(out["Data"]["Data"])
        chunks.append(z)

        first_ts=int(z["time"].min())
        if first_ts<=start_ts:
            break

        hit_ts=first_ts-86400
        time.sleep(1)

    allz=pd.concat(chunks,ignore_index=True).drop_duplicates("time")
    allz["Date"]=pd.to_datetime(allz["time"],unit="s",utc=True).dt.tz_convert(None).dt.normalize()
    allz=allz.set_index("Date").sort_index()
    closez=allz["close"].loc[pd.Timestamp(start):pd.Timestamp(end)]
    return closez

btc=ccgrab("BTC",cstart,cend)
eth=ccgrab("ETH",cstart,cend)

coinpx=pd.concat([btc,eth],axis=1)
coinpx.columns=["BTC","ETH"]
coinpx=coinpx.dropna()

coinpx.to_csv("btc_eth_daily_prices.csv")

print(coinpx.shape)
print(coinpx.index.min().date(),coinpx.index.max().date())
print(coinpx.head())

files.download("btc_eth_daily_prices.csv")

import numpy as np
import pandas as pd
from google.colab import files

coinpx=pd.read_csv("btc_eth_daily_prices.csv",index_col=0,parse_dates=True)

coinret=np.log(coinpx/coinpx.shift(1)).dropna()

coinret.to_csv("btc_eth_log_returns.csv")

print(coinret.shape)
print(coinret.index.min().date(),coinret.index.max().date())
print(coinret.head())

files.download("btc_eth_log_returns.csv")

import numpy as np
import pandas as pd
from google.colab import files

coinret=pd.read_csv("btc_eth_log_returns.csv",index_col=0,parse_dates=True)

leadbtc=coinret["BTC"].shift(-1)
leadeth=coinret["ETH"].shift(-1)

ytar=(leadbtc>leadeth).astype(float)
ytar[leadbtc.isna()|leadeth.isna()]=np.nan
ytar=ytar.dropna().astype(int)

ytab=pd.DataFrame({"y_next":ytar})
ytab.to_csv("btc_eth_target.csv")

print(ytab.shape)
print(ytab.index.min().date(),ytab.index.max().date())
print(ytab.head())
print(ytab["y_next"].value_counts(normalize=True).round(4))

files.download("btc_eth_target.csv")

import numpy as np
import pandas as pd
from google.colab import files

coinret=pd.read_csv("btc_eth_log_returns.csv",index_col=0,parse_dates=True)
ytab=pd.read_csv("btc_eth_target.csv",index_col=0,parse_dates=True)

lagbag=[]

for k in range(7):
    tmp=coinret.shift(k).copy()
    tmp.columns=[f"BTC_lag{k}",f"ETH_lag{k}"]
    lagbag.append(tmp)

xlag=pd.concat(lagbag,axis=1)

mlbox=xlag.join(ytab,how="inner").dropna()

mlbox.to_csv("btc_eth_7day_features_target.csv")

print(mlbox.shape)
print("n =",mlbox.shape[0])
print("p =",mlbox.shape[1]-1)
print(mlbox.head())

files.download("btc_eth_7day_features_target.csv")

import numpy as np
import pandas as pd
from google.colab import files

mlbox=pd.read_csv("btc_eth_7day_features_target.csv",index_col=0,parse_dates=True)

n=len(mlbox)
ntr=int(.70*n)
nva=int(.10*n)

trainbox=mlbox.iloc[:ntr].copy()
valbox=mlbox.iloc[ntr:ntr+nva].copy()
testbox=mlbox.iloc[ntr+nva:].copy()

trainbox.to_csv("btc_eth_train.csv")
valbox.to_csv("btc_eth_validation.csv")
testbox.to_csv("btc_eth_test.csv")

print("train:",trainbox.shape,trainbox.index.min().date(),trainbox.index.max().date())
print("validation:",valbox.shape,valbox.index.min().date(),valbox.index.max().date())
print("test:",testbox.shape,testbox.index.min().date(),testbox.index.max().date())

files.download("btc_eth_train.csv")
files.download("btc_eth_validation.csv")
files.download("btc_eth_test.csv")

import numpy as np
import pandas as pd

trainbox=pd.read_csv("btc_eth_train.csv",index_col=0,parse_dates=True)
valbox=pd.read_csv("btc_eth_validation.csv",index_col=0,parse_dates=True)
testbox=pd.read_csv("btc_eth_test.csv",index_col=0,parse_dates=True)

splitbag={"train":trainbox,"validation":valbox,"test":testbox}

rows=[]

for nm,df in splitbag.items():
    y=df["y_next"]
    p=df.shape[1]-1
    prop1=y.mean()
    baseerr=min(prop1,1-prop1)
    rows.append({
        "split":nm,
        "n":len(df),
        "p":p,
        "prop_y_1":prop1,
        "prop_y_0":1-prop1,
        "naive_baseline_error":baseerr
    })

balbox=pd.DataFrame(rows)

print(balbox.round(4))

"""The supervised dataset has p = 14 features in every split: seven BTC lag returns and seven ETH lag returns. The train split has n = 2648 with y = 1 proportion 0.5306. The validation split has n = 378 with y = 1 proportion 0.5661. The test split has n = 758 with y = 1 proportion 0.5567.

The dataset is not materially imbalanced because both classes are reasonably represented in each split. There is mild base-rate drift: the y = 1 rate rises from 53.06% in train to 56.61% in validation, then stays similar at 55.67% in test. The majority-class baseline error rates are 46.94% for train, 43.39% for validation, and 44.33% for test.
"""

from google.colab import files
files.upload()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import zero_one_loss
import warnings
warnings.filterwarnings("ignore")

trainbox=pd.read_csv("btc_eth_train.csv",index_col=0,parse_dates=True)
valbox=pd.read_csv("btc_eth_validation.csv",index_col=0,parse_dates=True)

xtr=trainbox.drop(columns=["y_next"]).values
ytr=trainbox["y_next"].values
xva=valbox.drop(columns=["y_next"]).values
yva=valbox["y_next"].values

cgrid=np.logspace(-4,4,60)
l1mix=.50

rows=[]

for cc in cgrid:
    logmod=make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty="elasticnet",solver="saga",l1_ratio=l1mix,C=cc,max_iter=20000,random_state=42)
    )
    logmod.fit(xtr,ytr)
    trpred=logmod.predict(xtr)
    vapred=logmod.predict(xva)
    rows.append({"C":cc,"train_error":zero_one_loss(ytr,trpred),"validation_error":zero_one_loss(yva,vapred)})

logscan=pd.DataFrame(rows)
bestrow=logscan.loc[logscan["validation_error"].idxmin()]
bestC=float(bestrow["C"])

best_logit=make_pipeline(
    StandardScaler(),
    LogisticRegression(penalty="elasticnet",solver="saga",l1_ratio=l1mix,C=bestC,max_iter=20000,random_state=42)
)
best_logit.fit(xtr,ytr)

plt.figure(figsize=(9,5.5))
plt.plot(logscan["C"],logscan["train_error"],lw=1.5,label="Train error")
plt.plot(logscan["C"],logscan["validation_error"],lw=1.8,label="Validation error")
plt.axvline(bestC,color="red",ls="--",lw=1.1,label=f"best C = {bestC:.3g}")
plt.xscale("log")
plt.xlabel("Inverse regularization strength C")
plt.ylabel("Error rate")
plt.title(f"Elastic-net logistic regression validation curve, l1_ratio = {l1mix}")
plt.grid(alpha=.3,which="both")
plt.legend()
plt.tight_layout()
plt.savefig("logit_elasticnet_validation_curve.png",dpi=150,bbox_inches="tight")
plt.show()

logscan.to_csv("logit_elasticnet_validation_scan.csv",index=False)

print("best C =",round(bestC,6))
print("best validation error =",round(float(bestrow["validation_error"]),4))
print("train error at best C =",round(float(bestrow["train_error"]),4))

files.download("logit_elasticnet_validation_curve.png")
files.download("logit_elasticnet_validation_scan.csv")

"""I fit an elastic-net logistic regression using the standardized 7-day BTC and ETH lag-return features. I fixed l1_ratio = 0.50 and tuned the inverse regularization strength C on the validation set, using classification error rate as the validation metric. The best value was C = 2.1826, giving a validation error rate of 0.4259 and a training error rate of 0.4566. The validation curve is fairly flat once C becomes moderately large, suggesting that the model is not very sensitive to the exact regularization level. The selected model slightly improves on the validation naive baseline error of 0.4339."""

!pip install -q xgboost

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import zero_one_loss
from xgboost import XGBClassifier
from google.colab import files
import warnings
warnings.filterwarnings("ignore")

trainbox=pd.read_csv("btc_eth_train.csv",index_col=0,parse_dates=True)
valbox=pd.read_csv("btc_eth_validation.csv",index_col=0,parse_dates=True)

xtr=trainbox.drop(columns=["y_next"]).values
ytr=trainbox["y_next"].values
xva=valbox.drop(columns=["y_next"]).values
yva=valbox["y_next"].values

rfrows=[]

for depth in [2,3,4,5,7,None]:
    for leaf in [5,10,20]:
        rfmod=RandomForestClassifier(n_estimators=400,max_depth=depth,min_samples_leaf=leaf,random_state=42,n_jobs=-1)
        rfmod.fit(xtr,ytr)
        rfrows.append({
            "model":"RandomForest",
            "max_depth":"None" if depth is None else depth,
            "min_samples_leaf":leaf,
            "train_error":zero_one_loss(ytr,rfmod.predict(xtr)),
            "validation_error":zero_one_loss(yva,rfmod.predict(xva))
        })

rfscan=pd.DataFrame(rfrows)
rfbest=rfscan.loc[rfscan["validation_error"].idxmin()]

def cleandepth(z):
    return None if str(z)=="None" else int(float(z))

rfdepth=cleandepth(rfbest["max_depth"])
rfleaf=int(rfbest["min_samples_leaf"])

best_rf=RandomForestClassifier(n_estimators=400,max_depth=rfdepth,min_samples_leaf=rfleaf,random_state=42,n_jobs=-1)
best_rf.fit(xtr,ytr)

xgbrows=[]

for depth in [1,2,3]:
    for rate in [.02,.05,.10]:
        for ntree in [100,200,300]:
            xgbmod=XGBClassifier(n_estimators=ntree,max_depth=depth,learning_rate=rate,subsample=.85,colsample_bytree=.85,eval_metric="logloss",random_state=42,n_jobs=-1)
            xgbmod.fit(xtr,ytr)
            xgbrows.append({
                "model":"XGBoost",
                "max_depth":depth,
                "learning_rate":rate,
                "n_estimators":ntree,
                "train_error":zero_one_loss(ytr,xgbmod.predict(xtr)),
                "validation_error":zero_one_loss(yva,xgbmod.predict(xva))
            })

xgbscan=pd.DataFrame(xgbrows)
xgbbest=xgbscan.loc[xgbscan["validation_error"].idxmin()]

best_xgb=XGBClassifier(n_estimators=int(xgbbest["n_estimators"]),max_depth=int(xgbbest["max_depth"]),learning_rate=float(xgbbest["learning_rate"]),subsample=.85,colsample_bytree=.85,eval_metric="logloss",random_state=42,n_jobs=-1)
best_xgb.fit(xtr,ytr)

rfscan.to_csv("rf_validation_scan.csv",index=False)
xgbscan.to_csv("xgb_validation_scan.csv",index=False)

print("Best random forest:")
print(rfbest)

print("\nBest XGBoost:")
print(xgbbest)

files.download("rf_validation_scan.csv")
files.download("xgb_validation_scan.csv")

"""I lightly tuned Random Forest over maximum tree depth and minimum leaf size, and XGBoost over tree depth, learning rate, and number of trees. The best Random Forest used max_depth = 3 and min_samples_leaf = 5, giving a validation error rate of 0.4206. The best XGBoost model used max_depth = 3, learning_rate = 0.02, and 300 trees, giving a validation error rate of 0.4259. Random Forest performed slightly better on validation error, while XGBoost had a much lower training error, suggesting more overfitting."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import zero_one_loss
from google.colab import files
import copy,random,warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

trainbox=pd.read_csv("btc_eth_train.csv",index_col=0,parse_dates=True)
valbox=pd.read_csv("btc_eth_validation.csv",index_col=0,parse_dates=True)

xtr=trainbox.drop(columns=["y_next"]).values
ytr=trainbox["y_next"].values.reshape(-1,1)
xva=valbox.drop(columns=["y_next"]).values
yva=valbox["y_next"].values.reshape(-1,1)

scalebox=StandardScaler()
xtrz=scalebox.fit_transform(xtr)
xvaz=scalebox.transform(xva)

xtr_t=torch.tensor(xtrz,dtype=torch.float32)
ytr_t=torch.tensor(ytr,dtype=torch.float32)
xva_t=torch.tensor(xvaz,dtype=torch.float32)
yva_t=torch.tensor(yva,dtype=torch.float32)

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(14,8),
            nn.ReLU(),
            nn.Linear(8,1)
        )
    def forward(self,x):
        return self.net(x)

nnmod=TinyNet()
param_count=sum(p.numel() for p in nnmod.parameters() if p.requires_grad)

lossfn=nn.BCEWithLogitsLoss()
opt=torch.optim.Adam(nnmod.parameters(),lr=.005)

best_loss=np.inf
best_epoch=0
best_state=None
patience=60
bad=0
hist=[]

for ep in range(1,2001):
    nnmod.train()
    opt.zero_grad()
    out=nnmod(xtr_t)
    loss=lossfn(out,ytr_t)
    loss.backward()
    opt.step()

    nnmod.eval()
    with torch.no_grad():
        trloss=lossfn(nnmod(xtr_t),ytr_t).item()
        valoss=lossfn(nnmod(xva_t),yva_t).item()

    hist.append({"epoch":ep,"train_loss":trloss,"validation_loss":valoss})

    if valoss<best_loss:
        best_loss=valoss
        best_epoch=ep
        best_state=copy.deepcopy(nnmod.state_dict())
        bad=0
    else:
        bad+=1

    if bad>=patience:
        break

nnmod.load_state_dict(best_state)
nnmod.eval()

with torch.no_grad():
    trprob=torch.sigmoid(nnmod(xtr_t)).numpy().ravel()
    vaprob=torch.sigmoid(nnmod(xva_t)).numpy().ravel()

trpred=(trprob>=.5).astype(int)
vapred=(vaprob>=.5).astype(int)

trerr=zero_one_loss(ytr.ravel(),trpred)
vaerr=zero_one_loss(yva.ravel(),vapred)

hbox=pd.DataFrame(hist)
hbox.to_csv("nn_training_history.csv",index=False)

print("parameter count =",param_count)
print("10% of training n =",.10*len(trainbox))
print("best epoch =",best_epoch)
print("best validation loss =",round(best_loss,6))
print("train error =",round(trerr,4))
print("validation error =",round(vaerr,4))

files.download("nn_training_history.csv")

"""I trained a feedforward neural network with one hidden layer of 8 ReLU units. With 14 input features, the exact number of trainable parameters is (14×8+8)+(8×1+1)=129. This satisfies the constraint because 10% of the training sample is 264.8, and 129 < 264.8. I used the validation set for early stopping; training stopped with the best validation loss at epoch 13. The final neural network had a training error rate of 0.4585 and a validation error rate of 0.4418."""

!pip install -q xgboost

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import copy,random,warnings
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import zero_one_loss
from xgboost import XGBClassifier
from google.colab import files
warnings.filterwarnings("ignore")

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

trainbox=pd.read_csv("btc_eth_train.csv",index_col=0,parse_dates=True)
valbox=pd.read_csv("btc_eth_validation.csv",index_col=0,parse_dates=True)
testbox=pd.read_csv("btc_eth_test.csv",index_col=0,parse_dates=True)

xtr=trainbox.drop(columns=["y_next"]).values
ytr=trainbox["y_next"].values
xva=valbox.drop(columns=["y_next"]).values
yva=valbox["y_next"].values
xte=testbox.drop(columns=["y_next"]).values
yte=testbox["y_next"].values

def errhit(mod,x,y):
    return zero_one_loss(y,mod.predict(x))

def baseerr(y):
    p1=np.mean(y)
    return min(p1,1-p1)

logC=2.182645
logit=make_pipeline(
    StandardScaler(),
    LogisticRegression(penalty="elasticnet",solver="saga",l1_ratio=.50,C=logC,max_iter=20000,random_state=42)
)
logit.fit(xtr,ytr)

rfmod=RandomForestClassifier(n_estimators=400,max_depth=3,min_samples_leaf=5,random_state=42,n_jobs=-1)
rfmod.fit(xtr,ytr)

xgbmod=XGBClassifier(n_estimators=300,max_depth=3,learning_rate=.02,subsample=.85,colsample_bytree=.85,eval_metric="logloss",random_state=42,n_jobs=-1)
xgbmod.fit(xtr,ytr)

scalebox=StandardScaler()
xtrz=scalebox.fit_transform(xtr)
xvaz=scalebox.transform(xva)
xtez=scalebox.transform(xte)

xtr_t=torch.tensor(xtrz,dtype=torch.float32)
ytr_t=torch.tensor(ytr.reshape(-1,1),dtype=torch.float32)
xva_t=torch.tensor(xvaz,dtype=torch.float32)
yva_t=torch.tensor(yva.reshape(-1,1),dtype=torch.float32)
xte_t=torch.tensor(xtez,dtype=torch.float32)

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(14,8),nn.ReLU(),nn.Linear(8,1))
    def forward(self,x):
        return self.net(x)

nnmod=TinyNet()
param_count=sum(p.numel() for p in nnmod.parameters() if p.requires_grad)

lossfn=nn.BCEWithLogitsLoss()
opt=torch.optim.Adam(nnmod.parameters(),lr=.005)

best_loss=np.inf
best_state=None
bad=0
patience=60
best_epoch=0

for ep in range(1,2001):
    nnmod.train()
    opt.zero_grad()
    loss=lossfn(nnmod(xtr_t),ytr_t)
    loss.backward()
    opt.step()

    nnmod.eval()
    with torch.no_grad():
        valoss=lossfn(nnmod(xva_t),yva_t).item()

    if valoss<best_loss:
        best_loss=valoss
        best_state=copy.deepcopy(nnmod.state_dict())
        best_epoch=ep
        bad=0
    else:
        bad+=1

    if bad>=patience:
        break

nnmod.load_state_dict(best_state)
nnmod.eval()

def nnerr(x_t,y):
    with torch.no_grad():
        prob=torch.sigmoid(nnmod(x_t)).numpy().ravel()
    pred=(prob>=.5).astype(int)
    return zero_one_loss(y,pred)

tabbox=pd.DataFrame({
    "Train error":[
        errhit(logit,xtr,ytr),
        errhit(rfmod,xtr,ytr),
        errhit(xgbmod,xtr,ytr),
        nnerr(xtr_t,ytr),
        baseerr(ytr)
    ],
    "Validation error":[
        errhit(logit,xva,yva),
        errhit(rfmod,xva,yva),
        errhit(xgbmod,xva,yva),
        nnerr(xva_t,yva),
        baseerr(yva)
    ],
    "Test error":[
        errhit(logit,xte,yte),
        errhit(rfmod,xte,yte),
        errhit(xgbmod,xte,yte),
        nnerr(xte_t,yte),
        baseerr(yte)
    ]
},index=["Elastic-net logistic","Random forest","XGBoost","Neural network","Naive baseline"])

tabbox=tabbox.round(4)

print("Neural network parameter count =",param_count)
print("Best NN epoch =",best_epoch)
print(tabbox)

tabbox.to_csv("model_comparison_error_rates.csv")
files.download("model_comparison_error_rates.csv")


Random Forest had the lowest validation error at 0.4206, slightly beating the validation naive baseline of 0.4339. Elastic-net logistic regression and XGBoost tied on validation error at 0.4259, while the neural network performed worse on validation. On the test set, however, none of the four models clearly beat the naive baseline of 0.4433; the neural network came closest with a test error of 0.4472. XGBoost had the lowest training error but the worst test error, which suggests overfitting. Overall, the lagged BTC and ETH returns provide only weak predictive signal for next-day BTC versus ETH outperformance.
"""
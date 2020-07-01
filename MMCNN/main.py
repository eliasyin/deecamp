from util import Util
import pandas as pd 


def main():
    util = Util()
    input_pickle = '/home/yincongfeng/deecamp/seed.pkl'
    util.run(input_pickle, lr=0.01, epochs=1000, early_stop=50, seed=20)
    # 将训练过程保存到csv文件，用于绘制分析
    df = pd.DataFrame({'train loss': util.train_loss_log,
                        'val loss': util.val_loss_log,
                        'val accuracy': util.val_acc_log})
    df.to_csv('log.csv')




if __name__ == '__main__':
    main()
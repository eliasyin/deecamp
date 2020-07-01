from util import run


def main():

    input_pickle = '/home/yincongfeng/deecamp/seed.pkl'
    run(input_pickle, lr=0.01, epochs=1000, early_stop=50, seed=None)



if __name__ == '__main__':
    main()
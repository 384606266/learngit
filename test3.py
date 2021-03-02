import threading
import time
import datetime

def fun1(x):                #函数1：阶乘
    result = 1
    for i in range(x):
        j = i+1
        result *= j
    print("%d的阶乘为：%d" % (x,result))

def fun2(x):                #函数2：累加
    result = 0
    for i in range(x):
        j = i+1
        result += j
    print("%d的累加为：%d" % (x,result))

def main():
    print("Main thread start!")
    threads = []
    thread_lock = threading.Lock()

    for m in (0,1):         #创建线程
        thread = threading.Thread(target=fun1,args=(m+5,))
        threads.append(thread)
        thread2 = threading.Thread(target=fun2,args=(m+5,))
        threads.append(thread2)

    for m in range(4):      #运行线程，且加上锁
        thread_lock.acquire()
        threads[m].start()
        thread_lock.release()
    
    for m in range(4):
        threads[m].join()

    print("Main thread finish!")


if __name__=="__main__":
    main()
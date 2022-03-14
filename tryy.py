import yueshu
if __name__ == '__main__':
    env = yueshu.Maze()
    env.after(200,yueshu.find_way())
    env.mainloop()



from tkinter import *
from tkinter.filedialog import askopenfile
from tkVideoPlayer import TkinterVideo

window = Tk()
window.title("Tkinter Play Videos in Video Player")
window.geometry("1200x750")
window.configure(bg="orange red")


current_course_frame = LabelFrame(window,bd=2,bg="white",relief=RIDGE,text="Current Course",font=("verdana",12,"bold"),fg="navyblue")
current_course_frame.place(x=10,y=170,width=635,height=550)

def open_file():
    file = askopenfile(mode='r', filetypes=[
                       ('Video Files', ["*.mp4"])])
    if file is not None:
        global filename
        filename = file.name
        global videoplayer
        videoplayer = TkinterVideo(master=current_course_frame, scaled=True)
        videoplayer.load(r"{}".format(filename))
        print(filename)
        videoplayer.pack(expand=True, fill="both")
        videoplayer.play()



def playAgain():
    print(filename)
    videoplayer.play()

def StopVideo():
    print(filename)
    videoplayer.stop()
 
def PauseVideo():
    print(filename)
    videoplayer.pause()

 
# center this label

current_course_frame = LabelFrame(window,bd=2,bg="white",relief=RIDGE,text="Current Course",font=("verdana",12,"bold"),fg="navyblue")
current_course_frame.place(x=10,y=170,width=635,height=550)

lbl1 = Label(current_course_frame, text="Tkinter Video Player", bg="orange red",
             fg="white", font="none 24 bold")
lbl1.config(anchor=CENTER)
lbl1.pack()


course_frame = LabelFrame(window,bd=2,bg="white",relief=RIDGE,text="Current Course",font=("verdana",12,"bold"),fg="navyblue")
course_frame.place(x=10,y=5,width=635,height=150)


openbtn = Button(course_frame, text='Open', command=lambda: open_file())
openbtn.pack(side=TOP, pady=2)

playbtn = Button(course_frame, text='Play Video', command=lambda: playAgain())
playbtn.pack(side=TOP, pady=3)

stopbtn = Button(course_frame, text='Stop Video', command=lambda: StopVideo())
stopbtn.pack(side=TOP, padx=4)

pausebtn = Button(course_frame, text='Pause Video', command=lambda: PauseVideo())
pausebtn.pack(side=TOP, padx=5)


window.mainloop()




# from tkinter import *
# from tkvideo import tkvideo

# root = Tk()
# my_label = Label(root)
# my_label.pack()
# player = tkvideo("polyp.mp4", my_label, loop = 1, size = (1280,720))
# player.play()

# root.mainloop()


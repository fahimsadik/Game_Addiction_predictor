import aiproject
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import joblib  # Import joblib directly
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns




class ScrollableFormApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Questionnaire")

        self.header_label = tk.Label(self.master, text="never=1, rarely=2, sometimes=3, often=4, very often=5",
                                      font=("Arial", 10, "bold"))
        self.header_label.pack(pady=10)

        self.questions = [
            "Did you think about playing a game all day long?",
            "Did you spend much free time on games?",
            "Have you felt addicted to a game?",
            "Did you play longer than intended?",
            "Did you spend increasing amounts of time on games?",
            "Were you unable to stop once you started playing?",
            "Did you play games to forget about real life?",
            "Have you played games to Realise Stress?",
            "Have you played games to feel better?",
            "Have you failed when trying to reduce game time?",
            "Have others unsuccessfully tried to reduce your game use?",
            "Have you felt bad and angry when you were unable to play?",
            "Have you become stressed when unable to play?",
            "Have you neglected others (e.g., family, friends) because you were playing games?",
            "Did you have fights with others (e.g., family, friends) over your time spent on games?",
            "Have you lost an important relationship because of your gaming activity?",
            "Have you deceived any of your family members or others because the amount of your gaming activity?",
            "Has your time on games caused sleep deprivation?",
            "Have you lost interests in previous hobbies and other entertainment activities as a result of your engagement with the game?",
            "Have you neglected other important activities (e.g., school, work, sports) to play games?",
            "Are you experiencing neck or back pain?",
            "Do you have orthopedic (joint and muscle) problems?",
            "Do you experience eyesight problems?",
            "Do you experience hearing problems?",
            # Add more questions here
        ]
        self.answers = []

        self.create_scrollable_frame()

    def create_scrollable_frame(self):
        self.scrollable_frame = tk.Frame(self.master)
        self.scrollable_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.scrollable_frame)
        self.scrollbar = ttk.Scrollbar(self.scrollable_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame_inner = tk.Frame(self.canvas)

        self.scrollable_frame_inner.bind("<Configure>",
                                         lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame_inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.create_question_widgets()

    def create_question_widgets(self):
        for i, question in enumerate(self.questions):
            question_label = tk.Label(self.scrollable_frame_inner, text=question)
            question_label.grid(row=i, column=0, padx=10, pady=10, sticky='w')

            answer_var = tk.IntVar()
            answer_var.set(0)  # Default value

            for option in range(1, 6):  # Options from 1 to 5
                radio_button = tk.Radiobutton(self.scrollable_frame_inner, text=str(option), variable=answer_var,
                                               value=option)
                radio_button.grid(row=i, column=option, padx=10, pady=5, sticky='w')

            self.answers.append(answer_var)

        self.submit_button = tk.Button(self.master, text="Submit", command=self.submit_answer)
        self.submit_button.pack(pady=10)

        # Prediction Result Label
        self.prediction_label = tk.Label(self.scrollable_frame_inner, text="", font=("Arial", 12))
        self.prediction_label.grid(row=len(self.questions), column=0, columnspan=6, pady=10, sticky='w')

        # Heatmap
        self.show_graph_button = tk.Button(self.master, text="Show Heatmap", command=self.show_heatmap)
        self.show_graph_button.pack(pady=10)

    def submit_answer(self):
        unanswered = False
        for i, answer_var in enumerate(self.answers):
            answer = answer_var.get()
            if answer == 0:
                unanswered = True
                messagebox.showwarning("Warning", f"Please answer question {i + 1} before submitting.")
                break

        if not unanswered:
            # Prepare the answers for prediction
            user_answers = [answer_var.get() for answer_var in self.answers]
            # Trim the user answers to the first 20 questions
            user_answers_first_model = user_answers[:20]
            user_answers_second_model = [user_answers[i-1] for i in [1, 5, 7, 8, 15, 16, 17, 19]]
            user_answers_third_model = user_answers[20:]

            # Load models
            model_addiction = joblib.load('Addiction')
            model_mental = joblib.load('mental')
            model_physical = joblib.load('physic')

            # Predictions
            addiction_prediction = model_addiction.predict([user_answers_first_model])
            mental_prediction = model_mental.predict([user_answers_second_model])
            physical_prediction = model_physical.predict([user_answers_third_model])

            # Convert predictions to "Yes" or "No"
            addiction_result = "Yes" if addiction_prediction[0] == 1 else "No"
            mental_result = "Yes" if mental_prediction[0] == 1 else "No"
            physical_result = "Yes" if physical_prediction[0] == 1 else "No"

            # Display predictions in the root window
            result_message = "RESULT\n\n"
            prediction_message = (
                f"Addiction prediction: {addiction_result}\n"
                f"Addiction accuracy: {aiproject.accuracy}, "
                f"Addiction F1 score: {aiproject.f1}, "
                f"Addiction precision: {aiproject.precision}, "
                f"Addiction Recall: {aiproject.recall}, "
                f"Addiction Confusion Matrix: {aiproject.conf_matrix}\n\n"
                f"Mental health prediction: {mental_result}\n"
                f"Mental accuracy: {aiproject.accuracy5}, "
                f"Mental F1 score: {aiproject.accuracy5}, "
                f"Mental precision: {aiproject.accuracy5}, "
                f"Mental Recall: {aiproject.accuracy5}, "
                f"Mental Confusion Matrix: {aiproject.accuracy5}\n\n"
                f"Physical problems prediction: {physical_result}\n"
                f"Physical accuracy: {aiproject.accuracy6}, "
                f"Physical F1 score: {aiproject.f16}, "
                f"Physical precision: {aiproject.precision6}, "
                f"Physical Recall: {aiproject.recall6}, "
                f"Physical Confusion Matrix: {aiproject.conf_matrix6}"
                
            )
            self.prediction_label.config(text=result_message + prediction_message)

    def show_heatmap(self):
        fig, ax = plt.subplots(figsize=(20,20))
        sns.heatmap(aiproject.df.corr(), annot=True, cmap='RdYlGn', ax=ax)
        ax.set_title('Correlation Heatmap')
        plt.tight_layout()

                # Convert the Matplotlib figure to a tkinter-compatible format and display it in the application
        self.canvas_heatmap = FigureCanvasTkAgg(fig, master=self.master)
        self.canvas_heatmap.draw()
        self.canvas_heatmap.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    

    def on_configure(self, event):
        self.canvas_hist.configure(scrollregion=self.canvas_hist.bbox("all"))


def main():
    root = tk.Tk()
    root.geometry("1000x800")
    app = ScrollableFormApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

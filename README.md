Data Sources
====================================

The data for this chatbot has been taken from the various souces. The main source for the data Regarding AIT is taken from the various web pages of the AIT web site and it related web sites. 


The various web pages from AIT website includes:

* Faculty Information:
    -   https://oldweb.ait.ac.th/people1/faculty/
    -   https://www.asdu.ait.ac.th/interimcodes/faculty/FacultyBySchool.cfm?SchoolID=3
    -   https://ait.ac.th/about/meet-our-faculty/

* Courses Informations:
    -   https://oldweb.ait.ac.th/admissions/eligibility/course-catalogue/set_main/


Overall Description and historical data is taken from Wikipedia:
*   https://en.wikipedia.org/wiki/Asian_Institute_of_Technology


**Faculty Information** **Courses Information** and **Wikipedia Information** is added in a single text file - **AIT-Wiki.txt**


The Data is also taken from the BROCHURE PDF downloaded from AIT Website - **AIT-BROCHURE.pdf**


Additional Information is also taken from the paper *A Unique Contribution to Engineering Education in Asia: The Asian Insititute of Technology*
The paper is also downloaded as pdf form the https://www.ijee.ie/articles/Vol09-4/090405.PDF - **090405.pdf**





Task 2. Analysis and Problem Solving
=========================================

The output from the Langchain pipeline shows that the model cansuccessfully generated a response to the general AIT related questions.

- The model correctly identified the relevant information from the input documents and provided a concise response which indicates that the model accurately understood the context and extracted the necessary information.

- The response directly addresses the question asked, providing relevant answer with less irrelevant information. This demonstrates the model's is focusing on the key details alright.

- The response is clear and easy to understand, with no grammatical blunders ambiguities in most of the queries.

- The response covers almost all the necessary information provided in the data to answer the questions without omitting any relevant details.

Overall, based on the its output, model appears to perform decent in retrieving information related to AIT and answering user queries accurately and effectively. 

### Possible Issues
In few of the cases model response provides information, which is unrelated to the question. This indicates that the model may not accurately discern the relevance of information within the input documents or may generate responses that are not directly related to the user's query.

This indicates that the model is not mature enough for the advance contextual understanding and filtering mechanisms.

Some of the reasons for the above issues may be:

- The model may not fully understand the context of the input question or documents, leading it to provide irrelevant responses.

- The model has not been fine-tuned very perfectly for the task of retrieving relevant information related to the Asian Institute of  Technology (AIT). Fine-tuning helps the model learn task-specific patterns and improve its performance on that particular task.

- Sometimes model can face challenges due to ambiguous or unclear information, making it challenging for the model to determine the most relevant response.

- The model is fine tuned on a data that does not incorporates all the possible queries and may lack in certain context of inquiries related to AIT, leading to suboptimal performance in providing relevant responses.


Task 3. Chatbot - Web Application
========================================

##### **Overview**
This is flask web application consists of chat windows which allows users to input query text. After generating the answer from the AIT-GPT based on the given query the chat window also provides with the response/answer.

- This web application consists of single web page - Home Page(*index.html*).
   * Home Page: 
   ![alt text](./app/static/home-page.png?raw=true)
   Here user can input query text and our chatbot will generate the appropriate response regarding AIT.

The application uses **chatbot.py**, which has the model that can provide gentle and informative answers
  


#### **Sample chats**

- Query 1:
    ![alt text](./app/static/chat-query-1.png?raw=true)

- Query 2:
    ![alt text](./app/static/chat-query-2.png?raw=true)

- Query 3:
    ![alt text](./app/static/chat-query-3.png?raw=true)

- Query 4:
    ![alt text](./app/static/chat-query-4.png?raw=true)

##### **Running the Application**
The Flask application is run using python app.py in the terminal, and the web interface can be accessed at http://127.0.0.1:5000/.


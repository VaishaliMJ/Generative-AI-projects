"""-------------------------------------------------------------------------------------
Problem Statement   :  Create a SQLite database to store and manage data for a chatbot application.
                
Author              :   Vaishali M. Jorwekar  

Date                :   17 Nov 2026
                
-------------------------------------------------------------------------------------"""
import sqlite3
import os

###############################################################################
#   Function        :   createTable
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Creates a SQLite Table  to store Students information in the database
#   Author          :   Vaishali M Jorwekar
#   Date            :   17 Nov 2016
###############################################################################
def createTable():
    
    tableInfo="""Create table if not exists STUDENT(
        STUDENT_NAME VARCHAR(25),
        CLASS VARCHAR(15),
        SECTION VARCHAR(15),
        MARKS INT)"""
    return tableInfo   
###############################################################################
#   Function        :   addStudentRecords
#   Input Params    :   cursor
#   Output Params   :   None
#   Description     :   Adds student records to the STUDENT table in the database
#   Author          :   Vaishali M Jorwekar
#   Date            :   17 Nov 2016
###############################################################################
def addStudentRecords(cursor):
#   Insert the student data into the STUDENT table
    cursor.execute('''Insert Into STUDENT Values ('Vaishali','Data Science','A',95)''')
    cursor.execute('''Insert Into STUDENT Values ('Riya','Logic Building','B',88)''')
    cursor.execute('''Insert Into STUDENT Values ('Aryan','PPA','A',92)''')
    cursor.execute('''Insert Into STUDENT Values ('Rahul','Data Science','B',85)''')
    cursor.execute('''Insert Into STUDENT Values ('Raghav','AI','B',95)''')
 
###############################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
#   Date            :   17 Nov 2016
###############################################################################
def main():
    #   Create a connection to the SQLite database with name 'Student.db'.
    
    connection = sqlite3.connect('Student.db')
    
    #   Create a cursor object to execute SQL commands
    cursor=connection.cursor()

    
    
    #   Create a table to store Students information
    tableInfo=createTable()
    
    #   Execute the SQL command to create the table
    cursor.execute(tableInfo)
    
    #   Insert the student data into the STUDENT table
    addStudentRecords(cursor)

    
    #   Display All the records
    print("Displaying all the records from the STUDENT table:")
    data = cursor.execute('''Select * from Student''')
    for row in data:
        print(row)
    
    #   Commit the changes and close the connection
    connection.commit()
    connection.close()
    
    print("SQLite database and table created successfully.")

###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    
    main()


import sqlite3


user_id = '123'
request = 'res'
connection = sqlite3.connect('db.sqlite3')
cursor = connection.cursor()
cursor.execute("INSERT INTO requests (user_id, request_text) VALUES ('%s', '%s')"%(user_id, request))

connection.commit()
connection.close()
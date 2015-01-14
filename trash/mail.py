import smtplib
def send_message(message):
    fromaddr = 'elitenect@gmail.com'
    toaddrs  = 'xf1280@gmail.com'
    msg = message
    # Credentials (if needed)
    username = 'elitenect@gmail.com'
    password = 'elitenect123'
    # The actual mail send
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.starttls()
    server.login(username,password)
    server.sendmail(fromaddr, toaddrs, msg)
    server.quit()
    
if __name__ == '__main__':
    send_message('helloworld')
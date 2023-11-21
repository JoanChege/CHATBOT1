document.getElementById("current-time").innerHTML = new Date().toLocaleTimeString()
  function sendMessage() {
    var messageInput = document.getElementById("message-input");
    var message = messageInput.value.trim();
    if (message) {
      var chatLog = document.getElementById("chat-log");
      var userMessageContainer = document.createElement("div")
      var userMessage = document.createElement("li");
      var userContainer = document.createElement("div")
      var userNameTime = document.createElement("div")
      var userName = document.createElement("div");
      var userMessageTimestamp = document.createElement("div");
      var userMessageText = document.createElement("div");
      var userIcon = document.createElement("div")

      userNameTime.classList.add("username-timestamp")
      userMessageTimestamp.textContent = new Date().toLocaleTimeString();
      userMessageTimestamp.classList.add("timestamp");
      userName.textContent = "User  "
      userName.classList.add("username")

      userMessageText.textContent = message;
      userMessageText.classList.add("user-input")

      userNameTime.appendChild(userName)
      userNameTime.appendChild(userMessageTimestamp)


      userContainer.classList.add("user-data-container")
      userContainer.appendChild(userNameTime)
      userContainer.appendChild(userMessageText)

      userIcon.classList.add("user-icon")
      userMessage.appendChild(userContainer);

      userMessage.classList.add("user-message");
      userMessageContainer.classList.add("user-message-container")
      userMessageContainer.appendChild(userMessage)
      userMessageContainer.appendChild(userIcon)

      chatLog.appendChild(userMessageContainer);

      chatLog.scrollTop += 500;
      messageInput.value = "";

      var xhr = new XMLHttpRequest();
      xhr.open("POST", "/get");
      xhr.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
      xhr.onreadystatechange = function () {
        if (this.readyState === XMLHttpRequest.DONE && this.status === 200) {
          var botMessageContainer = document.createElement("div")
          var botMessage = document.createElement("li");
          var botContainer = document.createElement("div")
          var botNameTime = document.createElement("div")
          var botName = document.createElement("div")
          var botMessageTimestamp = document.createElement("div");
          var botMessageText = document.createElement("div");
          var botIcon = document.createElement("div")

          botNameTime.classList.add("username-timestamp")
          botName.textContent = "Budbot"
          botName.classList.add("botname")
          botMessageTimestamp.textContent = new Date().toLocaleTimeString();
          botMessageTimestamp.classList.add("timestamp");

          botMessageText.textContent = this.responseText;
          botMessageText.classList.add("bot-output")

          botNameTime.appendChild(botName)
          botNameTime.appendChild(botMessageTimestamp)

          userContainer.classList.add("bot-data-container")
          botMessage.appendChild(botNameTime);
          botMessage.appendChild(botMessageText);

          botIcon.classList.add("bot-icon")

          botMessage.appendChild(botContainer)
          botMessage.classList.add("bot-message");

          botMessageContainer.classList.add("bot-message-container")
          botMessageContainer.appendChild(botIcon)
          botMessageContainer.appendChild(botMessage)

          chatLog.appendChild(botMessageContainer);
          chatLog.scrollTop += 500;
        }
        else if (this.readyState === XMLHttpRequest.DONE && this.status !== 200) {
          console.log('There was an error.');
        }
      };
      xhr.send("message=" + encodeURIComponent(message));
    }
  }

  const btnS = document.getElementById("message-input")

  btnS.addEventListener("keydown", (e) => {
    if (e.keyCode === 13) {
      e.preventDefault()
      console.log("pressed")

    }
  })
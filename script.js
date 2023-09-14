// Get a reference to the fetch button
const fetchButton = document.getElementById('fetch-button');

// Add an event listener to the button
fetchButton.addEventListener('click', function () {
    // Get the article title from the input field
    const articleTitle = document.getElementById('url-input').value;

    // Call the isFakeNews function to check if it's potentially fake news
    const result = isFakeNews(articleTitle);

    // Get a reference to the result message element
    const resultMessage = document.getElementById('result-message');

    // Display the result message on the webpage
    if (result) {
        resultMessage.textContent = "This news article may be fake.";
    } else {
        resultMessage.textContent = "This news article appears to be legitimate.";
    }
});

// Function for fake news detection
function isFakeNews(articleTitle) {
    // Define an array of trigger words or phrases that might indicate fake news
    const triggerWords = [
        'conspiracy',
        'hoax',
        'fake',
        'unverified',
        'rumor',
        'scam',
    ];

    // Convert the article title to lowercase for case-insensitive matching
    const lowercaseTitle = articleTitle.toLowerCase();

    // Check if any of the trigger words are present in the article title
    for (const word of triggerWords) {
        if (lowercaseTitle.includes(word)) {
            return true; // It's potentially fake news
        }
    }

    return false; // It's not flagged as fake news
}

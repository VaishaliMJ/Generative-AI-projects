USE questionbank;
CREATE DATABASE IF NOT EXISTS questionbank;


CREATE TABLE IF NOT EXISTS  questions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    topic VARCHAR(50) NOT NULL,
    complexity ENUM('Easy', 'Medium', 'Hard') NOT NULL,
    question TEXT NOT NULL
);

create database if not exists msu_lost_found;

-- Optional local setup only:
-- Replace change-me with your own MySQL password before running.
ALTER USER 'root'@'localhost' IDENTIFIED BY 'change-me';

GRANT ALL PRIVILEGES ON msu_lost_found.* TO 'root'@'localhost';
FLUSH PRIVILEGES;

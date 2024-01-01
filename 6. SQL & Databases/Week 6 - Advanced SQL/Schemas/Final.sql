-- 1-1. List the school names, community names and average attendance for communities with hardship index of 98

SELECT 
P.NAME_OF_SCHOOL, P.AVERAGE_STUDENT_ATTENDANCE, P.COMMUNITY_AREA_NAME, C.HARDSHIP_INDEX 
FROM CHICAGO_PUBLIC_SCHOOLS P 
LEFT OUTER JOIN CHICAGO_CENSUS_DATA C ON P.COMMUNITY_AREA_NUMBER = C.COMMUNITY_AREA_NUMBER 
WHERE C.HARDSHIP_INDEX = 98;

-- 1-2. List all crimes that took place at a school. include case number crime type, and community name

SELECT R.CASE_NUMBER, R.PRIMARY_TYPE, C.COMMUNITY_AREA_NAME, R.LOCATION_DESCRIPTION 
FROM CHICAGO_CRIME_DATA R 
LEFT JOIN CHICAGO_CENSUS_DATA C ON R.COMMUNITY_AREA_NUMBER = C.COMMUNITY_AREA_NUMBER 
WHERE R.LOCATION_DESCRIPTION LIKE 'SCHOOL%';

-- 2-1. Create a view that displays.

CREATE VIEW CHICAGO_SCHOOLS AS 

             SELECT 
             NAME_OF_SCHOOL AS School_Name, 
             SAFETY_ICON AS Safety_Rating, 
             FAMILY_INVOLVEMENT_ICON AS Family_Rating, 
             ENVIRONMENT_ICON AS Environment_Rating, 
             INSTRUCTION_ICON AS Instruction_Rating, 
             LEADERS_ICON AS Leaders_Rating,
             TEACHERS_ICON AS Teachers_Rating
             FROM CHICAGO_PUBLIC_SCHOOLS
             LIMIT 10; 

SELECT * FROM CHICAGO_SCHOOLS;

SELECT school_name, leaders_rating FROM CHICAGO_SCHOOLS 

-- 3-1. Write a query to create or replace a stored procedure called UPDATE_LEADERS_SCORE that takes in_school_ID parameter as integer and in_Leader_Score paramater as integer. use #SET TERMINATOR @

CREATE PROCEDURE UPDATE_LEADERS_SCORE

-- 3-2. Inside the previous stored procedure, write a SQL statement to update the Leaders_Score field in the public schools table for the school identified by in_school_id to the value in the in_leader_score parameter

CREATE PROCEDURE UPDATE_LEADERS_SCORE
       (IN SCHOOL_ID INT, IN LEADERS_RATING INT)

CALL UPDATE_LEADERS_SCORE(609947,79)

-- 3-3. Inside the stored procedure, write a SQL IF statement to update the leaders_icon field in the public schools table for the school identified by in_school_id using the following information.

DELIMITER @

CREATE PROCEDURE UPDATE_LEADERS_SCORE
   (IN SCHOOL_ID INT, IN LEADERS_RATING INT)  

       BEGIN

         IF LEADERS_RATING >79 THEN
           UPDATE CHICAGO_SCHOOLS
           SET LEADERS_RATING = 'Very Strong'
           WHERE ID = SCHOOL_ID;
         ELSEIF LEADERS_RATING > 59 AND LEADERS_RATING < 80 THEN
           UPDATE CHICAGO_SCHOOLS
           SET LEADERS_RATING = 'STRONG'
           WHERE ID = SCHOOL_ID;
         ELSEIF LEADERS_RATING > 39 AND LEADERS_RATING < 60 THEN
           UPDATE CHICAGO_SCHOOLS
           SET LEADERS_RATING = 'Average'
           WHERE ID = SCHOOL_ID;
         ELSEIF LEADERS_RATING > 19 AND LEADERS_RATING <40 THEN
           UPDATE CHICAGO_SCHOOLS
           SET LEADERS_RATING = 'Weak'
           WHERE ID = SCHOOL_ID;
         ELSEIF LEADERS_RATING < 20 THEN
           UPDATE CHICAGO_SCHOOLS
           SET LEADERS_RATING = 'Very Weak'
           WHERE ID = SCHOOL_ID;

       END @
DELIMITER;

-- 3-4. Write a query to call the stored procedure, passing a valid school ID and a leader score of 50, to check that the procedure works as expected

CALL UPDATE_LEADERS_SCORE(609947,79)

-- 4-1 update stored procedure definition. Add a generic ELSE clause to the IF statement that rolls back to the current work if the score did not fit any of the preceding categories.
-- # Hint: you can add an ELSE clause to the IF statement which will only run if none of the previous conditions have been met.

DELIMITER @

CREATE PROCEDURE UPDATE_LEADERS_SCORE
   (IN SCHOOL_ID INT, IN LEADERS_RATING INT)  

       BEGIN

         IF LEADERS_RATING >79 THEN
           UPDATE CHICAGO_SCHOOLS
           SET LEADERS_RATING = 'Very Strong'
           WHERE ID = SCHOOL_ID;
         ELSEIF LEADERS_RATING > 59 AND LEADERS_RATING < 80 THEN
           UPDATE CHICAGO_SCHOOLS
           SET LEADERS_RATING = 'STRONG'
           WHERE ID = SCHOOL_ID;
         ELSEIF LEADERS_RATING > 39 AND LEADERS_RATING < 60 THEN
           UPDATE CHICAGO_SCHOOLS
           SET LEADERS_RATING = 'Average'
           WHERE ID = SCHOOL_ID;
         ELSEIF LEADERS_RATING > 19 AND LEADERS_RATING <40 THEN
           UPDATE CHICAGO_SCHOOLS
           SET LEADERS_RATING = 'Weak'
           WHERE ID = SCHOOL_ID;
         ELSEIF LEADERS_RATING < 20 THEN
           UPDATE CHICAGO_SCHOOLS
           SET LEADERS_RATING = 'Very Weak'
           WHERE ID = SCHOOL_ID;
        ELSE
          ROLLBACK WORK;
        END IF;

       END @
DELIMITER;

-- 4-2. Update stored procedure definition again. Add a statement to commit the current unit of work at the end of the procedure.
-- # Hint: Remember that as soon as any code inside the IF/ELSE IF/ELSE statement completes, processing resumes after the END IF, so you can add your commit code there.
-- # Write and run one query to check that the updated stored procedure works as expected when you use a valid score of 38.
-- # Write and Run another query to check that the updated stored procedure works as expected when you use an invalid score of 101.

DELIMITER @

CREATE PROCEDURE UPDATE_LEADERS_SCORE
   (IN SCHOOL_ID INT, IN LEADERS_RATING INT)  

       BEGIN

         IF LEADERS_RATING >79 THEN
           UPDATE CHICAGO_SCHOOLS
           SET LEADERS_RATING = 'Very Strong'
           WHERE ID = SCHOOL_ID;
         ELSEIF LEADERS_RATING > 59 AND LEADERS_RATING < 80 THEN
           UPDATE CHICAGO_SCHOOLS
           SET LEADERS_RATING = 'STRONG'
           WHERE ID = SCHOOL_ID;
         ELSEIF LEADERS_RATING > 39 AND LEADERS_RATING < 60 THEN
           UPDATE CHICAGO_SCHOOLS
           SET LEADERS_RATING = 'Average'
           WHERE ID = SCHOOL_ID;
         ELSEIF LEADERS_RATING > 19 AND LEADERS_RATING <40 THEN
           UPDATE CHICAGO_SCHOOLS
           SET LEADERS_RATING = 'Weak'
           WHERE ID = SCHOOL_ID;
         ELSEIF LEADERS_RATING < 20 THEN
           UPDATE CHICAGO_SCHOOLS
           SET LEADERS_RATING = 'Very Weak'
           WHERE ID = SCHOOL_ID;
        ELSE
          ROLLBACK WORK;
        END IF;

        COMMIT WORK;

       END @
DELIMITER;

CALL UPDATE_LEADERS_SCORE(609947,79)

 select school_id, LEADERS_ICON from CHICAGO_PUBLIC_SCHOOLS_DATA where school_id = 609947;
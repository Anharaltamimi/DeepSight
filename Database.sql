-- Database.sql
CREATE DATABASE IF NOT EXISTS `deepsight_db` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE `deepsight_db`;

-- Drop tables if they exist to ensure a clean setup
DROP TABLE IF EXISTS Diagnoses;
DROP TABLE IF EXISTS Patients;
DROP TABLE IF EXISTS Doctors;

-- Create Doctors table with new columns (Email, Experience, Hospital)
CREATE TABLE Doctors (
  ID INT AUTO_INCREMENT PRIMARY KEY,
  Doctor_ID VARCHAR(20) UNIQUE NOT NULL,
  Doctor_Name VARCHAR(100) NOT NULL,
  Password VARCHAR(100) NOT NULL,
  Specialization VARCHAR(50) NOT NULL,
  Phone_Num VARCHAR(15),
  Email VARCHAR(100),
  Experience INT,
  Hospital VARCHAR(100),
  Profile_Image VARCHAR(300)
);

-- Add index on Doctor_Name
ALTER TABLE Doctors
  ADD INDEX idx_doctor_name (Doctor_Name);

-- Create Patients table
CREATE TABLE Patients (
  ID INT AUTO_INCREMENT PRIMARY KEY,
  Patient_ID VARCHAR(20) UNIQUE NOT NULL,
  Patient_Name VARCHAR(100) NOT NULL,
  Gender VARCHAR(10) NOT NULL,
  Date_Of_Birth DATE NOT NULL
);

-- Create Diagnoses table with updated Date_Of_Scan and UNIQUE constraint on Patient_ID
CREATE TABLE Diagnoses (
  ID INT AUTO_INCREMENT PRIMARY KEY,
  Patient_Name VARCHAR(100) NOT NULL,
  Patient_ID VARCHAR(20)  NOT NULL,
  Doctor_Name VARCHAR(100) NOT NULL,
  Date_Of_Scan DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  Diagnosis_Result VARCHAR(200),
  FOREIGN KEY (Patient_ID) REFERENCES Patients(Patient_ID),
  FOREIGN KEY (Doctor_Name) REFERENCES Doctors(Doctor_Name)
);

-- Add Report_File column to Diagnoses
ALTER TABLE Diagnoses ADD COLUMN Report_File VARCHAR(255);

-- Insert unique Doctors data (merging both files, avoiding duplicates)
INSERT INTO Doctors (Doctor_ID, Doctor_Name, Password, Specialization, Phone_Num, Email, Experience, Hospital) VALUES
('D0001', 'Dr.Sara Saud', 'pass001', 'Ophthalmology', '0503981497', 'anhar.altamimi24@gmail.com', 10, 'King Khalid Eye Hospital'),
('D0002', 'Dr.Yara Alarjani', 'pass002', 'Ophthalmology', '0509577607', 'yara.alarjani@example.com', 8, 'King Khalid Eye Hospital'),
('D0003', 'Dr.Samyah Souliman', 'pass003', 'Ophthalmology', '0509388110', 'samyah.souliman@example.com', 12, 'King Khalid Eye Hospital'),
('D0004', 'Dr.Nouf Aluthaimeen', 'pass004', 'Ophthalmology', '0509388650', 'nouf.aluthaimeen@example.com', 7, 'King Khalid Eye Hospital'),
('D0005', 'Dr.Raghad Alanzi', 'pass005', 'Ophthalmology', '0509388168', 'raghad.alanzi@example.com', 9, 'King Khalid Eye Hospital');


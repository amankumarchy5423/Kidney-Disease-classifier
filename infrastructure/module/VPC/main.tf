resource "aws_vpc" "project_vpc" {
  cidr_block = vars.vpc_cidr
  tags = {
    Name = "kidney-Disease-VPC"
  }
}

resource "aws_subnet" "project_subnet" {
  vpc_id = aws_vpc.project_vpc.id
  n = vars.subnet.length
  cidr_block = 

}
img = flipud(imread("00_background.png"));
figure
image(img)
set(gca, 'YDir', 'normal');
hold on
% grid on

xmin = 400;
xmax = 1125;
ymin = 5;
ymax = 936;

% ymin = 550;
% ymax = 900;
% xmin = 710;
% xmax = 800;

show_const = true;

if ~show_const
    xs = xmin:10:xmax;
    ys = ymin:10:ymax;
    for i = 1:length(xs)
        for j = 1:length(ys)
            scatter(xs(i),ys(j), 5, 'r', 'filled')
        end
    end

else
    % Circle 1 (center circle)
    p1 = [835 470];
    p2 = [760 410];
    p3 = [730 470];
    
    [h1,k1,r1] = solve_circle(p1,p2,p3);
    plot_cirlce(h1,k1,r1);

    % Circle 2 (upper left)
    p1 = [730 605];
    p2 = [700 515];
    p3 = [595 485];
    
    [h2,k2,r2] = solve_circle(p1,p2,p3);
    plot_cirlce(h2,k2,r2);

    % Circle 3 (upper right)
    p1 = [835 575];
    p2 = [850 530];
    p3 = [925 500];
    
    [h3,k3,r3] = solve_circle(p1,p2,p3);
    plot_cirlce(h3,k3,r3);

    % Circle 4 (lower right)
    p1 = [850 305];
    p2 = [865 395];
    p3 = [925 425];
    
    [h4,k4,r4] = solve_circle(p1,p2,p3);
    plot_cirlce(h4,k4,r4);

    % Circle 5 (lower left)
    p1 = [640 410];
    p2 = [700 395];
    p3 = [730 350];
    
    [h5,k5,r5] = solve_circle(p1,p2,p3);
    plot_cirlce(h5,k5,r5);

    % Ellipse 1 (Lower Median)
    p1 = [788 356];
    p2 = [815 311];
    p3 = [815 254];
    p4 = [785 305];
    p5 = [788 293];
    x = [p1(1);p2(1);p3(1);p4(1);p5(1);];
    y = [p1(2);p2(2);p3(2);p4(2);p5(2);];
    ellip1= fit_ellipse(x,y);
    plot_ellipse(ellip1.X0_in,ellip1.Y0_in,ellip1.a,ellip1.b, -ellip1.phi);

    % Ellipse 2 (Right Median)
    p1 = [934 472];
    p2 = [886 457];
    p3 = [937 457];
    p4 = [955 460];
    p5 = [991 475];
    x = [p1(1);p2(1);p3(1);p4(1);p5(1);];
    y = [p1(2);p2(2);p3(2);p4(2);p5(2);];
    ellip2= fit_ellipse(x,y);
    plot_ellipse(ellip2.X0_in,ellip2.Y0_in,ellip2.a,ellip2.b, -ellip2.phi);

    % Ellipse 3 (Left Median)
    p1 = [483 466];
    p2 = [486 460];
    p3 = [546 445];
    p4 = [633 445];
    p5 = [567 463];
    x = [p1(1);p2(1);p3(1);p4(1);p5(1);];
    y = [p1(2);p2(2);p3(2);p4(2);p5(2);];
    ellip3= fit_ellipse(x,y);
    plot_ellipse(ellip3.X0_in,ellip3.Y0_in,ellip3.a,ellip3.b, -ellip3.phi);

    % Ellipse 4 (Top Median)
    p1 = [776 562];
    p2 = [764 745];
    p3 = [752 718];
    p4 = [739 859];
    p5 = [737 890];
    x = [p1(1);p2(1);p3(1);p4(1);p5(1);];
    y = [p1(2);p2(2);p3(2);p4(2);p5(2);];
    ellip4= fit_ellipse(x,y);
    plot_ellipse(ellip4.X0_in,ellip4.Y0_in,ellip4.a,ellip4.b, -ellip4.phi);

    % Line 1 (Left, Up)
    p1 = [420 505];
    p2 = [595 485];
    [m1,b1] = solve_line(p1,p2);
    plot_line(m1,b1, p1(1),p2(1));

    % Line 2 (Up, Left)
    p1 = [640 935];
    p2 = [730 605];
    [m2,b2] = solve_line(p1,p2);
    plot_line(m2,b2, p1(1),p2(1));

    % Line 3 (Up, Right)
    p1 = [800 795];
    p2 = [835 575];
    [m3,b3] = solve_line(p1,p2);
    plot_line(m3, b3, p1(1),p2(1));
    
    % Line 4 (Right, Up)
    p1 = [925 500];
    p2 = [1050 525];
    [m4,b4] = solve_line(p1,p2);
    plot_line(m4, b4, p1(1),p2(1));

    % Line 5 (Right, Lower)
    p1 = [925 425];
    p2 = [1100 465];
    [m5,b5] = solve_line(p1,p2);
    plot_line(m5, b5, p1(1),p2(1));

    % Line 6 (Lower, Right)
    p1 = [850 305];
    p2 = [920 5];
    [m6,b6] = solve_line(p1,p2);
    plot_line(m6, b6, p1(1),p2(1));

    % Line 7 (Lower, Left)
    p1 = [730 350];
    p2 = [850 5];
    [m7,b7] = solve_line(p1,p2);
    plot_line(m7, b7, p1(1),p2(1));

    % Line 8 (Left, Lower)
    p1 = [490 425];
    p2 = [640 410];
    [m8,b8] = solve_line(p1,p2);
    plot_line(m8, b8, p1(1),p2(1));

end

function [h,k,r] = solve_circle(p1, p2, p3)
    
    A = [p1(1) p1(2) 1; p2(1) p2(2) 1; p3(1) p3(2) 1];

    b = [-(p1(1)^2 + p1(2)^2);
         -(p2(1)^2 + p2(2)^2);
         -(p3(1)^2 + p3(2)^2)];
    x = A\b;

    h = -x(1)/2;
    k = -x(2)/2;
    r = sqrt(h^2 + k^2 - x(3));

end

function [m,b] = solve_line(p1, p2)
    m = (p2(2)-p1(2))/(p2(1)-p1(1));
    b = p1(2) - m*p1(1);
end

function [xs, ys] = plot_cirlce(h, k, r)
    theta = linspace(0,2*pi, 100);
    xs = r.*cos(theta) + h;
    ys = r.*sin(theta) + k;

    plot(xs,ys ,'r')
end

function [xs,ys] = plot_line(m, b, xmin, xmax)
    xs = linspace(xmin,xmax,100);
    ys = m.*xs + b;
    
    plot(xs,ys,'r')
end

function [xs, ys] = plot_ellipse(h,k,a,b,theta)
    t = linspace(0,2*pi,100);
    x = a*cos(t);
    y = b*sin(t);

    R = [cos(theta) -sin(theta); sin(theta) cos(theta)];

    ellipse_points = R*[x;y];

    xs = ellipse_points(1, :) + h;
    ys = ellipse_points(2, :) + k;

    plot(xs,ys,'r')
end